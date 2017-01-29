#include <vector>
#include <set>

#define TP_NOT_ZERO(v) (fabs(v) > 0.00001f)

#define TP_CPP_THREADS
#define TP_NUM_THREADS 8
#ifdef TP_CPP_THREADS
#include <thread>
#endif

static void *tp_alloc_void(size_t size)
{
#if defined(_MSC_VER)
	void *ret = _aligned_malloc(size, 8 * sizeof(float));
#else
	void *ret = aligned_alloc(8 * sizeof(float), size);
#endif
	memset(ret, 0, size);
	return ret;
	//return calloc(1, size);
}
#define tp_alloc(type, count) ((type*)tp_alloc_void(sizeof(type) * (count)))

static void tp_free(void *memory)
{
	//free(memory);
#if defined(_MSC_VER)
	_aligned_free(memory);
#else
	free(memory);
#endif
}

/*static void tp_float_line(float *data, int w, int h, int c, lm_vec2 p0, lm_vec2 p1, float r, float g, float b)
{
	int x0 = (int)(p0.x * w);
	int y0 = (int)(p0.y * h);
	int x1 = (int)(p1.x * w);
	int y1 = (int)(p1.y * h);
	int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
	int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
	int err = (dx > dy ? dx : -dy) / 2, e2;
	for (;;)
	{
		float *p = data + (y0 * w + x0) * c;
		p[0] = r; p[1] = g; p[2] = b;

		if (x0 == x1 && y0 == y1) break;
		e2 = err;
		if (e2 > -dx) { err -= dy; x0 += sx; }
		if (e2 <  dy) { err += dx; y0 += sy; }
	}
}*/

static inline bool tp_accumulate_texel(float *sums, int x, int y, float *data, int w, int h, int c)
{
	bool exists = false;
	for (int i = 0; i < c; i++)
	{
		float v = data[(y * w + x) * c + i];
		sums[i] += v;
		exists |= v > 0.0f;
	}
	return exists;
}

static void tp_fill_with_closest(int x, int y, float *data, int w, int h, int c, int depth = 2)
{
	assert(c <= 4);

	for (int i = 0; i < c; i++)
		if (data[(y * w + x) * c + i] > 0.0f)
			return;

	float sums[4] = {};
	int n = 0;

	if (x     > 0 && tp_accumulate_texel(sums, x - 1, y, data, w, h, c)) n++;
	if (x + 1 < w && tp_accumulate_texel(sums, x + 1, y, data, w, h, c)) n++;
	if (y     > 0 && tp_accumulate_texel(sums, x, y - 1, data, w, h, c)) n++;
	if (y + 1 < h && tp_accumulate_texel(sums, x, y + 1, data, w, h, c)) n++;

	if (!n && depth)
	{
		--depth;
		if (x > 0)
		{
			tp_fill_with_closest(x - 1, y, data, w, h, c, depth);
			if (tp_accumulate_texel(sums, x - 1, y, data, w, h, c)) n++;
		}
		if (x + 1 < w)
		{
			tp_fill_with_closest(x + 1, y, data, w, h, c, depth);
			if (tp_accumulate_texel(sums, x + 1, y, data, w, h, c)) n++;
		}
		if (y > 0)
		{
			tp_fill_with_closest(x, y - 1, data, w, h, c, depth);
			if (tp_accumulate_texel(sums, x, y - 1, data, w, h, c)) n++;
		}
		if (y + 1 < h)
		{
			tp_fill_with_closest(x, y + 1, data, w, h, c, depth);
			if (tp_accumulate_texel(sums, x, y + 1, data, w, h, c)) n++;
		}
	}

	if (n)
	{
		float ni = 1.0f / (float)n;
		for (int i = 0; i < c; i++)
			data[(y * w + x) * c + i] = sums[i] * ni;
	}
}

struct texel_t
{
	int16_t x, y;

	texel_t() { x = 0; y = 0; }
	texel_t(int16_t px, int16_t py) : x(px), y(py) {}
	bool operator<(const texel_t& rhs) const {
		return y < rhs.y || (y == rhs.y && x < rhs.x);
	}
};

struct bilinear_sample_t
{
	texel_t texels[4];
	float weights[4];
};
struct stitching_point_t
{
	bilinear_sample_t sides[2];
};

struct connected_set_t
{
	int16_t x_min, y_min, x_max, y_max;
	std::set<texel_t> texels;
	std::vector<stitching_point_t> stitchingPoints;
};

static void tp_connected_set_add(connected_set_t *set, stitching_point_t *point)
{
	for (int side = 0; side < 2; side++)
	{
		for (int texel = 0; texel < 4; texel++)
		{
			texel_t t = point->sides[side].texels[texel];
			set->x_min = t.x < set->x_min ? t.x : set->x_min;
			set->y_min = t.y < set->y_min ? t.y : set->y_min;
			set->x_max = t.x > set->x_max ? t.x : set->x_max;
			set->y_max = t.y > set->y_max ? t.y : set->y_max;
			set->texels.insert(t);
		}
	}

	set->stitchingPoints.push_back(*point);
}

static bool tp_connected_sets_intersect(connected_set_t *a, connected_set_t *b)
{
	// compare bounding boxes first
	if (a->x_min > b->x_max || b->x_min >= a->x_max ||
		a->y_min > b->y_max || b->y_min >= a->y_max)
		return false;

	// bounds intersect -> check each individual texel for intersection
	for (auto t = a->texels.begin(); t != a->texels.end(); ++t)
		if (b->texels.find(*t) != b->texels.end())
			return true;

	return false;
}

static void tp_connected_sets_in_place_merge(connected_set_t *dst, connected_set_t *src)
{
	// expand bounding box
	dst->x_min = src->x_min < dst->x_min ? src->x_min : dst->x_min;
	dst->y_min = src->y_min < dst->y_min ? src->y_min : dst->y_min;
	dst->x_max = src->x_max > dst->x_max ? src->x_max : dst->x_max;
	dst->y_max = src->y_max > dst->y_max ? src->y_max : dst->y_max;

	// insert src elements
	dst->texels.insert(src->texels.begin(), src->texels.end());
	dst->stitchingPoints.insert(dst->stitchingPoints.end(), src->stitchingPoints.begin(), src->stitchingPoints.end());
}

static void tp_connected_sets_add_seam(std::vector<connected_set_t>& connectedSets, lm_vec2 a0, lm_vec2 a1, lm_vec2 b0, lm_vec2 b1, float *data, int w, int h, int c)
{
	/*float r = rand() / (float)RAND_MAX;
	float g = rand() / (float)RAND_MAX;
	float b = rand() / (float)RAND_MAX;*/
	//tp_float_line(data, w, h, c, a0, a1, r, g, b);
	//tp_float_line(data, w, h, c, b0, b1, r, g, b);
	//tp_float_line(data, w, h, c, a0, a1, 1.0f, 0.0f, 0.0f);
	//tp_float_line(data, w, h, c, b0, b1, 0.0f, 1.0f, 0.0f);
	/*lm_vec2 s = lm_v2i(w, h);
	a0 = lm_mul2(a0, s);
	a1 = lm_mul2(a1, s);
	b0 = lm_mul2(b0, s);
	b1 = lm_mul2(b1, s);
	lm_vec2 ad = lm_sub2(a1, a0);
	lm_vec2 bd = lm_sub2(b1, b0);
	float l = lm_length2(ad);
	int iterations = (int)(l * 10.0f);
	float step = 1.0f / iterations;
	for (int i = 0; i <= iterations; i++)
	{
		float t = i * step;
		lm_vec2 a = lm_add2(a0, lm_scale2(ad, t));
		lm_vec2 b = lm_add2(b0, lm_scale2(bd, t));
		int ax = (int)roundf(a.x), ay = (int)roundf(a.y);
		int bx = (int)roundf(b.x), by = (int)roundf(b.y);
		for (int j = 0; j < c; j++)
		{
			float ac = data[(ay * w + ax) * c + j];
			float bc = data[(by * w + bx) * c + j];
			if (ac > 0.0f && bc > 0.0f)
			{
				float amount = (ac > 0.0f && bc > 0.0f) ? 0.5f : 1.0f;
				data[(ay * w + ax) * c + j] = data[(by * w + bx) * c + j] = amount * (ac + bc);
			}
		}
	}*/

	// opt

	lm_vec2 s = lm_v2i(w, h);
	a0 = lm_mul2(a0, s);
	a1 = lm_mul2(a1, s);
	b0 = lm_mul2(b0, s);
	b1 = lm_mul2(b1, s);
	lm_vec2 ad = lm_sub2(a1, a0);
	lm_vec2 bd = lm_sub2(b1, b0);
	float l = lm_length2(ad);
	int iterations = (int)(l * 5.0f);
	float step = 1.0f / iterations;

	connected_set_t currentSet;
	currentSet.x_min = w; currentSet.y_min = h;
	currentSet.x_max = 0; currentSet.y_max = 0;

	for (int i = 0; i <= iterations; i++)
	{
		float t = i * step;
		lm_vec2 a = lm_add2(a0, lm_scale2(ad, t));
		lm_vec2 b = lm_add2(b0, lm_scale2(bd, t));
		int16_t ax = (int16_t)roundf(a.x), ay = (int16_t)roundf(a.y);
		int16_t bx = (int16_t)roundf(b.x), by = (int16_t)roundf(b.y);
		float au = a.x - ax, av = a.y - ay, nau = 1.0f - au, nav = 1.0f - av;
		float bu = b.x - bx, bv = b.y - by, nbu = 1.0f - bu, nbv = 1.0f - bv;

		texel_t ta0(ax    , ay    );
		texel_t ta1(ax + 1, ay    );
		texel_t ta2(ax    , ay + 1);
		texel_t ta3(ax + 1, ay + 1);

		texel_t tb0(bx    , by    );
		texel_t tb1(bx + 1, by    );
		texel_t tb2(bx    , by + 1);
		texel_t tb3(bx + 1, by + 1);

		tp_fill_with_closest(ta0.x, ta0.y, data, w, h, c);
		tp_fill_with_closest(ta1.x, ta1.y, data, w, h, c);
		tp_fill_with_closest(ta2.x, ta2.y, data, w, h, c);
		tp_fill_with_closest(ta3.x, ta3.y, data, w, h, c);

		tp_fill_with_closest(tb0.x, tb0.y, data, w, h, c);
		tp_fill_with_closest(tb1.x, tb1.y, data, w, h, c);
		tp_fill_with_closest(tb2.x, tb2.y, data, w, h, c);
		tp_fill_with_closest(tb3.x, tb3.y, data, w, h, c);

		stitching_point_t sp;
		sp.sides[0].texels[0] = ta0;
		sp.sides[0].texels[1] = ta1;
		sp.sides[0].texels[2] = ta2;
		sp.sides[0].texels[3] = ta3;
		
		sp.sides[0].weights[0] = nau * nav;
		sp.sides[0].weights[1] = au * nav;
		sp.sides[0].weights[2] = nau * av;
		sp.sides[0].weights[3] = au * av;

		sp.sides[1].texels[0] = tb0;
		sp.sides[1].texels[1] = tb1;
		sp.sides[1].texels[2] = tb2;
		sp.sides[1].texels[3] = tb3;

		sp.sides[1].weights[0] = nbu * nbv;
		sp.sides[1].weights[1] = bu * nbv;
		sp.sides[1].weights[2] = nbu * bv;
		sp.sides[1].weights[3] = bu * bv;

		tp_connected_set_add(&currentSet, &sp);

		/*for (int j = 0; j < c; j++)
		{
			float ac = data[(ay * w + ax) * c + j];
			float bc = data[(by * w + bx) * c + j];
			if (ac > 0.0f && bc > 0.0f)
			{
				float amount = (ac > 0.0f && bc > 0.0f) ? 0.5f : 1.0f;
				data[(ay * w + ax) * c + j] = data[(by * w + bx) * c + j] = amount * (ac + bc);
			}
		}*/
	}

	int connectedSet = -1;
	for (int i = 0; i < connectedSets.size(); i++)
	{
		if (tp_connected_sets_intersect(&currentSet, &connectedSets[i]))
		{
			if (connectedSet == -1) // found a set that the edge is connected to -> add current edge to that set
			{
				tp_connected_sets_in_place_merge(&connectedSets[i], &currentSet);
				connectedSet = i;
			}
			else // found another set that the edge is connected to -> merge those sets
			{
				tp_connected_sets_in_place_merge(&connectedSets[connectedSet], &connectedSets[i]);
				connectedSets.erase(connectedSets.begin() + i);
				--i;
			}
		}
	}
	if (connectedSet == -1) // did not find a set that the edge is connected to -> make a new one
		connectedSets.push_back(currentSet);
}

static int tp_should_optimize(lm_vec3 *tria, lm_vec3 *trib)
{
	lm_vec3 n0 = lm_normalize3(lm_cross3(lm_sub3(tria[1], tria[0]), lm_sub3(tria[2], tria[0])));
	lm_vec3 n1 = lm_normalize3(lm_cross3(lm_sub3(trib[1], trib[0]), lm_sub3(trib[2], trib[0])));
	return lm_absf(lm_dot3(n0, n1)) > 0.99f; // TODO: make threshold an argument!
}

static int tp_texel_binary_search(texel_t *texels, int n, texel_t toFind)
{
	int n_half = n / 2;
	texel_t *center = texels + n_half;
	if (toFind.y == center->y && toFind.x == center->x)
		return n_half;
	if (n <= 1)
		return -1;
	if (toFind.y < center->y || (toFind.y == center->y && toFind.x < center->x))
		return tp_texel_binary_search(texels, n_half, toFind);
	else
	{
		int result = tp_texel_binary_search(center + 1, n - n_half - 1, toFind);
		return result == -1 ? -1 : n_half + 1 + result;
	}
}

#ifdef TP_CPP_THREADS
typedef struct
{
	const float *A;
	float *AtA;
	const int *sparseIndices;
	int maxRowIndices;
	int m, n;
	int inc;
} tp_matrix_At_times_A_data;

static void tp_matrix_At_times_A_lower_left(tp_matrix_At_times_A_data *data, int index)
{
	if (!data->sparseIndices || !data->maxRowIndices)
	{
		for (int k = 0; k < data->m; k++)
		{
			const float *srcPtr = data->A + k * data->n;
			for (int i = index; i < data->n; i += data->inc)
			{
				float v = data->A[k * data->n + i];
				if (TP_NOT_ZERO(v))
				{
					float *dstPtr = data->AtA + i * data->n;
					for (int j = 0; j <= i; j++, srcPtr++, dstPtr++)
						(*dstPtr) += v * (*srcPtr);
				}
			}
		}
	}
	else
	{
		for (int k = 0; k < data->m; k++)
		{
			const float *srcPtr = data->A + k * data->n;
			const int *indexPtr = data->sparseIndices + k * data->maxRowIndices;
			for (int i = index; i < data->n; i += data->inc)
			{
				float v = data->A[k * data->n + i];
				if (TP_NOT_ZERO(v))
				{
					float *dstPtr = data->AtA + i * data->n;
					for (int j = 0; j < data->maxRowIndices; j++)
					{
						int index = indexPtr[j];
						if (index < 0) break;
						dstPtr[index] += v * srcPtr[index];
					}
				}
			}
		}
	}
}
#endif

static void tp_matrix_At_times_A(const float *A, int m, int n, float *AtA, const int *sparseIndices = 0, int maxRowIndices = 0)
{
	//if (!AtAisZero)
	//	memset(AtA, 0, n * n * sizeof(float));

	// compute lower left triangle only since the result is symmetric
	tp_matrix_At_times_A_data data;
	data.A = A;
	data.AtA = AtA;
	data.sparseIndices = sparseIndices;
	data.maxRowIndices = maxRowIndices;
	data.m = m;
	data.n = n;

#if 0 //def TP_CPP_THREADS
	if (n >= 256 * TP_NUM_THREADS)
	{
		data.inc = TP_NUM_THREADS;

		std::thread threads[TP_NUM_THREADS];
		for (int i = 0; i < TP_NUM_THREADS; i++)
			threads[i] = std::thread(tp_matrix_At_times_A_lower_left, &data, i);

		for (int i = 0; i < TP_NUM_THREADS; i++)
			threads[i].join();
	}
	else
#endif
	{
		data.inc = 1;
		tp_matrix_At_times_A_lower_left(&data, 0);
	}

	// mirror lower left triangle to upper right
	//for (int i = 0; i < n; i++)
	//	for (int j = 0; j < i; j++)
	//		AtA[j * n + i] = AtA[i * n + j];
}

static void tp_matrix_At_times_b(const float *A, int m, int n, const float *b, float *Atb, const int *sparseIndices = 0, int maxRowIndices = 0)
{
	memset(Atb, 0, sizeof(float) * n);
	if (!sparseIndices || !maxRowIndices)
	{
		for (int j = 0; j < m; j++)
			for (int i = 0; i < n; i++)
				Atb[i] += A[j * n + i] * b[j];
	}
	else
	{
		for (int j = 0; j < m; j++)
		{
			const int *rowIndices = sparseIndices + j * maxRowIndices;
			for (int i = 0; i < maxRowIndices; i++)
			{
				int index = rowIndices[i];
				if (index < 0) break;
				Atb[index] += A[j * n + index] * b[j];
			}
		}
	}
}

static inline float tp_rsqrtf(float v)
{
	float r;
	_mm_store_ss(&r, _mm_rsqrt_ss(_mm_load_ss(&v)));
	return r;
}

static bool tp_matrix_cholesky_prepare(const float *A, int n, float *L, int *sparseIndicesOut = 0)
{
	// dense
	//for (int i = 0; i < n; i++)
	//{
	//	float *a = L + i * n;
	//	for (int j = 0; j <= i; j++)
	//	{
	//		float *b = L + j * n;
	//		float sum = A[i * n + j];// +(i == j ? 0.0001 : 0.0); // some regularization
	//		for (int k = 0; k < j; k++)
	//			sum -= a[k] * b[k];
	//		if (i > j)
	//			a[j] = sum / b[j];
	//		else // i == j
	//		{
	//			if (sum <= 0.0)
	//				return false;
	//			a[i] = sqrtf(sum);
	//		}
	//	}
	//}
	
	// sparse
	int *indices_i = sparseIndicesOut;
	if (!indices_i)
		indices_i = (int*)alloca(sizeof(int) * n);
	int *colCounts = (int*)alloca(sizeof(int) * n);
	float *invDiag = (float*)alloca(sizeof(float) * n);
	for (int i = 0; i < n; i++)
	{
		float *a = L + i * n;
		int index_i_count = 0;
		colCounts[i] = 0;
		for (int j = 0; j <= i; j++)
		{
			float *b = L + j * n;
			float sum = A[i * n + j]; // + (i == j ? 0.0001 : 0.0); // regularization
			for (int k = 0; k < index_i_count; k++)
			{
				int index_i = indices_i[k];
				sum -= a[index_i] * b[index_i];
			}

			if (i == j)
			{
				if (sum <= 0.0) return false;
				//_mm_store_ss(invDiag + i, _mm_rsqrt_ss(_mm_load_ss(&sum))); // invDiag[i] = 1.0f / sqrtf(sum)
				invDiag[i] = tp_rsqrtf(sum);
			}

			if (TP_NOT_ZERO(sum))
			{
				a[j] = sum * invDiag[j];
				indices_i[index_i_count++] = j;
				if (sparseIndicesOut && i != j)
				{
					int indexCol = colCounts[j]++;
					sparseIndicesOut[j * n + n - 1 - indexCol] = i;
				}
			}
			else
				a[j] = 0.0f;
		}
		if (sparseIndicesOut)
		{
			if (index_i_count)
				indices_i[--index_i_count] = -1;
			else
				indices_i[0] = -1;
			indices_i += n;
		}
	}

	if (sparseIndicesOut)
	{
		for (int i = 0; i < n; i++)
			sparseIndicesOut[i * n + n - 1 - colCounts[i]] = -1;
	}
	return true;
}

static void tp_matrix_cholesky_solve(const float *L, float *x, const float *b, int n, const int *LsparseIndices = 0)
{
	float *y = (float*)alloca(sizeof(float) * n); // TODO

	// L * y = b
	for (int i = 0; i < n; i++)
	{
		float sum = b[i];
		const float *row = L + i * n;
		if (LsparseIndices)
		{
			const int *rowIndices = LsparseIndices + i * n;
			for (int j = 0; j < i; j++)
			{
				int index = rowIndices[j];
				if (index < 0) break;
				sum -= row[index] * y[index];
			}
		}
		else
		{
			for (int j = 0; j < i; j++)
				sum -= row[j] * y[j];
		}
		y[i] = sum / row[i];
	}

	// L' * x = y
	for (int i = n - 1; i >= 0; i--)
	{
		float sum = y[i];
		const float *col = L + i;
		if (LsparseIndices)
		{
			const int *colIndices = LsparseIndices + i * n + n - 1;
			for (int j = 0; true; j--)
			{
				int index = colIndices[j];
				if (index < 0) break;
				sum -= col[index * n] * x[index];
			}
		}
		else
		{
			for (int j = n - 1; j > i; j--)
				sum -= col[j * n] * x[j];
		}
		x[i] = sum / col[i * n];
	}
}

typedef struct
{
	std::vector<connected_set_t> *connectedSets;
	float lambda;
	int w, h, c;
	float *data;
} tp_seams_t;

static void tp_optimize_seam(tp_seams_t *data, int setIndex)
{
	std::set<texel_t>& texels = (*data->connectedSets)[setIndex].texels;
	std::vector<stitching_point_t>& stitchingPoints = (*data->connectedSets)[setIndex].stitchingPoints;

	size_t m = stitchingPoints.size();
	size_t n = texels.size();
	if (n & 7) // align to AVX2 register size
		n += 8 - (n & 7);

	std::vector<texel_t> texelsFlat(texels.begin(), texels.end());
	float *A = tp_alloc(float, (m + n) * n);
	int *AsparseIndices = tp_alloc(int, (m + n) * 8);

	size_t r = 0;
	for (int i = 0; i < m; i++)
	{
		ptrdiff_t column0[4];
		ptrdiff_t column1[4];
		bool side0valid = false, side1valid = false;
		for (int k = 0; k < 4; k++)
		{
			texel_t t0 = stitchingPoints[i].sides[0].texels[k];
			texel_t t1 = stitchingPoints[i].sides[1].texels[k];
			column0[k] = tp_texel_binary_search(&texelsFlat[0], n, t0);
			column1[k] = tp_texel_binary_search(&texelsFlat[0], n, t1);

			if (column0[k] == -1) { side0valid = false; break; }
			if (column1[k] == -1) { side1valid = false; break; }

			// test for validity of stitching point
			for (int ci = 0; ci < data->c; ci++)
			{
				side0valid |= data->data[(t0.y * data->w + t0.x) * data->c + ci] > 0.0f;
				side1valid |= data->data[(t1.y * data->w + t1.x) * data->c + ci] > 0.0f;
			}
		}

		if (side0valid && side1valid)
		{
			for (int k = 0; k < 4; k++)
			{
				A[r * n + column0[k]] = stitchingPoints[i].sides[0].weights[k];
				AsparseIndices[r * 8 + k * 2 + 0] = column0[k];
				A[r * n + column1[k]] = -stitchingPoints[i].sides[1].weights[k];
				AsparseIndices[r * 8 + k * 2 + 1] = column1[k];
			}
			r++;
		}
	}

	m = r;

	// add error terms for deviation from original pixel value (scaled by lambda)
	for (int i = 0; i < n; i++)
	{
		A[(m + i) * n + i] = data->lambda;
		AsparseIndices[(m + i) * 8 + 0] = i;
		AsparseIndices[(m + i) * 8 + 1] = -1;
	}

	float *AtA = tp_alloc(float, n * n);

	//double start_0 = glfwGetTime();
	tp_matrix_At_times_A(A, m + n, n, AtA, AsparseIndices, 8);
	//printf("Took %.2fs.\n", glfwGetTime() - start_0); // 13s

	float *L = tp_alloc(float, n * n);
	int *LsparseIndices = tp_alloc(int, n * n);
	if (!tp_matrix_cholesky_prepare(AtA, n, L, LsparseIndices))
	{
		printf("Could not compute cholesky decomposition for set %d.\n", setIndex);
		tp_free(A);
		tp_free(AsparseIndices);
		tp_free(AtA);
		tp_free(L);
		tp_free(LsparseIndices);
		return;
	}

	//printf("%d, ", m);
	/*if (n > 100)
	{
		FILE *f = fopen("dump_L.txt", "wt");
		if (f)
		{
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j <= i; j++)
				{
					if (fabs(L[i * n + j]) > 0.00001f)
						fprintf(f, "%.2f, ", L[i * n + j]);
				}
				fprintf(f, "\n");
			}
			fclose(f);
			exit(0);
		}
	}*/

	float *b = tp_alloc(float, m + n);
	float *Atb = tp_alloc(float, n);
	float *x = tp_alloc(float, n);

	// solve each color channel independently
  	for (int ci = 0; ci < data->c; ci++)
	{
		auto texel = texels.begin();
		for (int i = 0; texel != texels.end(); i++, texel++)
			b[m + i] = data->lambda * data->data[(texel->y * data->w + texel->x) * data->c + ci];

		tp_matrix_At_times_b(A, m + n, n, b, Atb, AsparseIndices, 8);
		tp_matrix_cholesky_solve(L, x, Atb, n, LsparseIndices);

		// write out results
		texel = texels.begin();
		for (int i = 0; texel != texels.end(); i++, texel++)
			data->data[(texel->y * data->w + texel->x) * data->c + ci] = x[i];
	}

	tp_free(x);
	tp_free(Atb);
	tp_free(b);
	tp_free(L);
	tp_free(LsparseIndices);
	tp_free(AtA);
	tp_free(A);
	tp_free(AsparseIndices);
}

static void tp_optimize_seams(lm_vec3 *positions, lm_vec2 *texcoords, int vertices, float *data, int w, int h, int c, float lambda)
{
	lm_vec3 bbmin = lm_v3(FLT_MAX, FLT_MAX, FLT_MAX);
	lm_vec3 bbmax = lm_v3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	int *hashmap = tp_alloc(int, vertices * 2);
	for (int i = 0; i < vertices; i++)
	{
		bbmin = lm_min3(bbmin, positions[i]);
		bbmax = lm_max3(bbmax, positions[i]);
		hashmap[i * 2 + 0] = -1;
		hashmap[i * 2 + 1] = -1;
	}
	
	lm_vec3 bbscale = lm_v3(15.9f / bbmax.x, 15.9f / bbmax.y, 15.9f / bbmax.z);

	std::vector<connected_set_t> connectedSets;

	printf("Searching for sets of stitching points...\n");

	for (int i0 = 0; i0 < vertices; i0++)
	{
		int tri = i0 - (i0 % 3);
		int i1 = tri + ((i0 + 1) % 3);
		int i2 = tri + ((i0 + 2) % 3);
		lm_vec3 p = lm_mul3(lm_sub3(positions[i0], bbmin), bbscale);
		int hash = (281 * (int)p.x + 569 * (int)p.y + 1447 * (int)p.z) % (vertices * 2);
		while (hashmap[hash] >= 0)
		{
			int oi0 = hashmap[hash];
#define TP_EQUAL(a, b) lm_length3sq(lm_sub3(positions[a], positions[b])) < 0.0000001f
			if (TP_EQUAL(oi0, i0))
			{
				int otri = oi0 - (oi0 % 3);
				int oi1 = otri + ((oi0 + 1) % 3);
				int oi2 = otri + ((oi0 + 2) % 3);
				if (TP_EQUAL(oi1, i1) && tp_should_optimize(positions + tri, positions + otri))
					tp_connected_sets_add_seam(connectedSets, texcoords[i0], texcoords[i1], texcoords[oi0], texcoords[oi1], data, w, h, c);
				//else if (TP_EQUAL(oi1, i2) && tp_should_optimize(positions + tri, positions + otri)) // this will already be detected by the other side of the seam!
				//	tp_connected_sets_add_seam(connectedSets, texcoords[i0], texcoords[i2], texcoords[oi0], texcoords[oi1], data, w, h, c);
				else if (TP_EQUAL(oi2, i1) && tp_should_optimize(positions + tri, positions + otri))
					tp_connected_sets_add_seam(connectedSets, texcoords[i0], texcoords[i1], texcoords[oi0], texcoords[oi2], data, w, h, c);
			}
			if (++hash == vertices * 2)
				hash = 0;
		}
		hashmap[hash] = i0;
	}

	tp_free(hashmap);

	printf("Solving %d sets of stitching points...\n", connectedSets.size());
	double start = glfwGetTime();

	tp_seams_t seams;
	seams.connectedSets = &connectedSets;
	seams.lambda = lambda;
	seams.w = w;
	seams.h = h;
	seams.c = c;
	seams.data = data;

#if 0 //def TP_CPP_THREADS
	std::thread *threads = new std::thread[connectedSets.size()];

#define TP_THREAD_THRESHOLD 1024

	for (int setIndex = 0; setIndex < connectedSets.size(); setIndex++)
		if (connectedSets[setIndex].stitchingPoints.size() > TP_THREAD_THRESHOLD)
			threads[setIndex] = std::thread(tp_optimize_seam, &seams, setIndex);

	for (int setIndex = 0; setIndex < connectedSets.size(); setIndex++)
		if (connectedSets[setIndex].stitchingPoints.size() <= TP_THREAD_THRESHOLD)
			tp_optimize_seam(&seams, setIndex);

	for (int setIndex = 0; setIndex < connectedSets.size(); setIndex++)
		if (connectedSets[setIndex].stitchingPoints.size() > TP_THREAD_THRESHOLD)
			threads[setIndex].join();

	delete[] threads;
#else
	for (int setIndex = 0; setIndex < connectedSets.size(); setIndex++)
		tp_optimize_seam(&seams, setIndex);
#endif

	double duration = glfwGetTime() - start;
	printf("Took %.2fs.\n", duration); // 17s / 6s
}