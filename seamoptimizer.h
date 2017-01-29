#include <stdlib.h> // qsort
#include <stdio.h> // printf (TODO)
#include <string.h> // memcpy
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#ifndef SO_CALLOC
#include <malloc.h> // calloc, free, alloca (TODO)
#define SO_CALLOC calloc
#define SO_FREE free
#endif

//#define SO_CPP_THREADS
//#define SO_NUM_THREADS 8
//#ifdef SO_CPP_THREADS
//#include <thread>
//#endif


#define SO_EPSILON 0.00001f

#ifdef _DEBUG
#define SO_NOT_ZERO(v) (v > SO_EPSILON || v < -SO_EPSILON) // a lot faster in debug
#else
#define SO_NOT_ZERO(v) (fabsf(v) > SO_EPSILON) // faster in release
#endif


#ifdef SO_APPROX_RSQRT
#include "xmmintrin.h"
static inline float so_rsqrtf(float v)
{
	return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(v)));
}
#else
static inline float so_rsqrtf(float v)
{
	return 1.0f / sqrtf(v);
}
#endif

static inline float    so_minf      (float   a, float   b) { return a < b ? a : b; }
static inline float    so_maxf      (float   a, float   b) { return a > b ? a : b; }
static inline float    so_absf      (float   a           ) { return a < 0.0f ? -a : a; }

typedef struct so_vec2 { float x, y; } so_vec2;
static inline so_vec2  so_v2i       (int     x, int     y) { so_vec2 v = { (float)x, (float)y }; return v; }
static inline so_vec2  so_v2        (float   x, float   y) { so_vec2 v = { x, y }; return v; }
static inline so_vec2  so_add2      (so_vec2 a, so_vec2 b) { return so_v2(a.x + b.x, a.y + b.y); }
static inline so_vec2  so_sub2      (so_vec2 a, so_vec2 b) { return so_v2(a.x - b.x, a.y - b.y); }
static inline so_vec2  so_mul2      (so_vec2 a, so_vec2 b) { return so_v2(a.x * b.x, a.y * b.y); }
static inline so_vec2  so_scale2    (so_vec2 a, float   b) { return so_v2(a.x * b, a.y * b); }
static inline float    so_length2sq (so_vec2 a           ) { return a.x * a.x + a.y * a.y; }
static inline float    so_length2   (so_vec2 a           ) { return sqrtf(so_length2sq(a)); }

typedef struct so_vec3 { float x, y, z; } so_vec3;
static inline so_vec3  so_v3        (float   x, float   y, float   z) { so_vec3 v = { x, y, z }; return v; }
static inline so_vec3  so_sub3      (so_vec3 a, so_vec3 b) { return so_v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline so_vec3  so_mul3      (so_vec3 a, so_vec3 b) { return so_v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline so_vec3  so_scale3    (so_vec3 a, float   b) { return so_v3(a.x * b, a.y * b, a.z * b); }
static inline so_vec3  so_div3      (so_vec3 a, float   b) { return so_scale3(a, 1.0f / b); }
static inline so_vec3  so_min3      (so_vec3 a, so_vec3 b) { return so_v3(so_minf(a.x, b.x), so_minf(a.y, b.y), so_minf(a.z, b.z)); }
static inline so_vec3  so_max3      (so_vec3 a, so_vec3 b) { return so_v3(so_maxf(a.x, b.x), so_maxf(a.y, b.y), so_maxf(a.z, b.z)); }
static inline float    so_dot3      (so_vec3 a, so_vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline so_vec3  so_cross3    (so_vec3 a, so_vec3 b) { return so_v3(a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y); }
static inline float    so_length3sq (so_vec3 a           ) { return a.x * a.x + a.y * a.y + a.z * a.z; }
static inline float    so_length3   (so_vec3 a           ) { return sqrtf(so_length3sq(a)); }
static inline so_vec3  so_normalize3(so_vec3 a           ) { return so_div3(a, so_length3(a)); }

//#define SO_CHECK_FOR_MEMORY_LEAKS // check for memory leaks. don't use this in multithreaded code!

#ifdef SO_CHECK_FOR_MEMORY_LEAKS
static uint64_t so_allocated = 0;

static void *so_alloc_void(size_t size)
{
	void *memory = SO_CALLOC(1, size + sizeof(size_t));
	(*(size_t*)memory) = size;
	so_allocated += size;
	return (size_t*)memory + 1;
}
static void so_free(void *memory)
{
	size_t size = ((size_t*)memory)[-1];
	so_allocated -= size;
	SO_FREE(((size_t*)memory) - 1);
}
#else
static void *so_alloc_void(size_t size)
{
	return SO_CALLOC(1, size + sizeof(size_t));
}
static void so_free(void *memory)
{
	SO_FREE(memory);
}
#endif

#define so_alloc(type, count) ((type*)so_alloc_void(sizeof(type) * (count)))

static inline bool so_accumulate_texel(float *sums, int x, int y, float *data, int w, int h, int c)
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

static void so_fill_with_closest(int x, int y, float *data, int w, int h, int c, int depth = 2)
{
	assert(c <= 4);

	for (int i = 0; i < c; i++)
		if (data[(y * w + x) * c + i] > 0.0f)
			return;

	float sums[4] = {};
	int n = 0;

	if (x     > 0 && so_accumulate_texel(sums, x - 1, y, data, w, h, c)) n++;
	if (x + 1 < w && so_accumulate_texel(sums, x + 1, y, data, w, h, c)) n++;
	if (y     > 0 && so_accumulate_texel(sums, x, y - 1, data, w, h, c)) n++;
	if (y + 1 < h && so_accumulate_texel(sums, x, y + 1, data, w, h, c)) n++;

	if (!n && depth)
	{
		--depth;
		if (x > 0)
		{
			so_fill_with_closest(x - 1, y, data, w, h, c, depth);
			if (so_accumulate_texel(sums, x - 1, y, data, w, h, c)) n++;
		}
		if (x + 1 < w)
		{
			so_fill_with_closest(x + 1, y, data, w, h, c, depth);
			if (so_accumulate_texel(sums, x + 1, y, data, w, h, c)) n++;
		}
		if (y > 0)
		{
			so_fill_with_closest(x, y - 1, data, w, h, c, depth);
			if (so_accumulate_texel(sums, x, y - 1, data, w, h, c)) n++;
		}
		if (y + 1 < h)
		{
			so_fill_with_closest(x, y + 1, data, w, h, c, depth);
			if (so_accumulate_texel(sums, x, y + 1, data, w, h, c)) n++;
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
};

static inline int so_texel_cmp(const void *l, const void *r)
{
	const texel_t *lt = (const texel_t*)l;
	const texel_t *rt = (const texel_t*)r;
	if (lt->y < rt->y) return -1;
	if (lt->y > rt->y) return 1;
	if (lt->x < rt->x) return -1;
	if (lt->x > rt->x) return 1;
	return 0;
}

struct bilinear_sample_t
{
	texel_t texels[4];
	float weights[4];
};
struct stitching_point_t
{
	bilinear_sample_t sides[2];
};

struct texel_set_t
{
	texel_t *texels;
	uint32_t count;
	uint32_t capacity;
};

static inline uint32_t so_texel_hash(texel_t texel, uint32_t capacity)
{
	return (texel.y * 104173 + texel.x * 86813) % capacity;
}

static void so_texel_set_add(texel_set_t *set, texel_t *texels, int entries, int arrayLength = 0)
{
	if (set->count + entries > set->capacity * 3 / 4) // leave some free space to avoid having many collisions
	{
		int newCapacity = set->capacity > 64 ? set->capacity * 2 : 64;
		while (set->count + entries > newCapacity * 3 / 4)
			newCapacity *= 2;

		texel_t *newTexels = so_alloc(texel_t, newCapacity);

		for (int i = 0; i < newCapacity; i++)
			newTexels[i].x = -1;

		if (set->texels)
		{
			for (int i = 0; i < set->capacity; i++) // rehash all old texels
			{
				if (set->texels[i].x != -1)
				{
					uint32_t hash = so_texel_hash(set->texels[i], newCapacity);
					while (newTexels[hash].x != -1) // collisions
						hash = (hash + 1) % newCapacity;
					newTexels[hash] = set->texels[i];
				}
			}
			so_free(set->texels);
		}

		set->texels = newTexels;
		set->capacity = newCapacity;
	}

	if (arrayLength == 0)
		arrayLength = entries;

	for (int i = 0; i < arrayLength; i++)
	{
		if (texels[i].x != -1)
		{
			uint32_t hash = so_texel_hash(texels[i], set->capacity);
			while (set->texels[hash].x != -1) // collisions
			{
				if (set->texels[hash].x == texels[i].x && set->texels[hash].y == texels[i].y)
					break; // texel is already in the set
				hash = (hash + 1) % set->capacity;
			}

			if (set->texels[hash].x == -1)
			{
				set->texels[hash] = texels[i];
				set->count++;
			}
		}
	}
}

static bool so_texel_set_contains(texel_set_t *set, texel_t texel)
{
	uint32_t hash = so_texel_hash(texel, set->capacity);
	while (set->texels[hash].x != -1) // entries with same hash
	{
		if (set->texels[hash].x == texel.x && set->texels[hash].y == texel.y)
			return true; // texel is already in the set
		hash = (hash + 1) % set->capacity;
	}
	return false;
}

static void so_texel_set_free(texel_set_t *set)
{
	so_free(set->texels);
	*set = {0};
}

struct stitching_points_t
{
	stitching_point_t *points;
	uint32_t count;
	uint32_t capacity;
};
static void so_stitching_points_alloc(stitching_points_t *points, uint32_t n)
{
	points->points = so_alloc(stitching_point_t, n);
	points->capacity = n;
	points->count = 0;
}
static void so_stitching_points_free(stitching_points_t *points)
{
	so_free(points->points);
	*points = {0};
}
static void so_stitching_points_add(stitching_points_t *points, stitching_point_t *point)
{
	assert(points->count < points->capacity);
	points->points[points->count++] = *point;
}
static void so_stitching_points_append(stitching_points_t *points, stitching_points_t *other)
{
	stitching_point_t *newPoints = so_alloc(stitching_point_t, points->capacity + other->capacity);
	memcpy(newPoints, points->points, sizeof(stitching_point_t) * points->count);
	memcpy(newPoints + points->count, other->points, sizeof(stitching_point_t) * other->count);
	so_free(points->points);
	points->points = newPoints;
	points->capacity = points->capacity + other->capacity;
	points->count = points->count + other->count;
}

struct connected_set_t
{
	int16_t x_min, y_min, x_max, y_max;
	texel_set_t texels;
	stitching_points_t stitchingPoints;
	connected_set_t *next;
};

static void so_connected_set_alloc(connected_set_t *set, uint32_t stitchingPointCount)
{
	so_stitching_points_alloc(&set->stitchingPoints, stitchingPointCount);
}
static void so_connected_set_free(connected_set_t *set)
{
	so_texel_set_free(&set->texels);
	so_stitching_points_free(&set->stitchingPoints);
}

static void so_connected_set_add(connected_set_t *set, stitching_point_t *point)
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
		}
		so_texel_set_add(&set->texels, point->sides[side].texels, 4);
	}

	so_stitching_points_add(&set->stitchingPoints, point);
}

static bool so_connected_sets_intersect(connected_set_t *a, connected_set_t *b)
{
	// compare bounding boxes first
	if (a->x_min > b->x_max || b->x_min >= a->x_max ||
		a->y_min > b->y_max || b->y_min >= a->y_max)
		return false;

	// bounds intersect -> check each individual texel for intersection
	if (a->texels.capacity > b->texels.capacity) // swap so that we always loop over the smaller set
	{
		connected_set_t *tmp = a;
		a = b;
		b = tmp;
	}

	for (int i = 0; i < a->texels.capacity; i++)
		if (a->texels.texels[i].x != -1)
			if (so_texel_set_contains(&b->texels, a->texels.texels[i]))
				return true;
	return false;
}

static void so_connected_sets_in_place_merge(connected_set_t *dst, connected_set_t *src)
{
	// expand bounding box
	dst->x_min = src->x_min < dst->x_min ? src->x_min : dst->x_min;
	dst->y_min = src->y_min < dst->y_min ? src->y_min : dst->y_min;
	dst->x_max = src->x_max > dst->x_max ? src->x_max : dst->x_max;
	dst->y_max = src->y_max > dst->y_max ? src->y_max : dst->y_max;

	// insert src elements
	so_texel_set_add(&dst->texels, src->texels.texels, src->texels.count, src->texels.capacity);
	so_stitching_points_append(&dst->stitchingPoints, &src->stitchingPoints);
}

static void so_connected_sets_add_seam(connected_set_t **connectedSets, so_vec2 a0, so_vec2 a1, so_vec2 b0, so_vec2 b1, float *data, int w, int h, int c)
{
	so_vec2 s = so_v2i(w, h);
	a0 = so_mul2(a0, s);
	a1 = so_mul2(a1, s);
	b0 = so_mul2(b0, s);
	b1 = so_mul2(b1, s);
	so_vec2 ad = so_sub2(a1, a0);
	so_vec2 bd = so_sub2(b1, b0);
	float l = so_length2(ad);
	int iterations = (int)(l * 5.0f);
	float step = 1.0f / iterations;

	connected_set_t currentSet = {0};
	currentSet.x_min = w; currentSet.y_min = h;
	currentSet.x_max = 0; currentSet.y_max = 0;

	so_connected_set_alloc(&currentSet, iterations + 1);

	for (int i = 0; i <= iterations; i++)
	{
		float t = i * step;
		so_vec2 a = so_add2(a0, so_scale2(ad, t));
		so_vec2 b = so_add2(b0, so_scale2(bd, t));
		int16_t ax = (int16_t)roundf(a.x), ay = (int16_t)roundf(a.y);
		int16_t bx = (int16_t)roundf(b.x), by = (int16_t)roundf(b.y);
		float au = a.x - ax, av = a.y - ay, nau = 1.0f - au, nav = 1.0f - av;
		float bu = b.x - bx, bv = b.y - by, nbu = 1.0f - bu, nbv = 1.0f - bv;

		texel_t ta0 = { ax    , ay     };
		texel_t ta1 = { ax + 1, ay     };
		texel_t ta2 = { ax    , ay + 1 };
		texel_t ta3 = { ax + 1, ay + 1 };

		texel_t tb0 = { bx    , by     };
		texel_t tb1 = { bx + 1, by     };
		texel_t tb2 = { bx    , by + 1 };
		texel_t tb3 = { bx + 1, by + 1 };

		so_fill_with_closest(ta0.x, ta0.y, data, w, h, c);
		so_fill_with_closest(ta1.x, ta1.y, data, w, h, c);
		so_fill_with_closest(ta2.x, ta2.y, data, w, h, c);
		so_fill_with_closest(ta3.x, ta3.y, data, w, h, c);

		so_fill_with_closest(tb0.x, tb0.y, data, w, h, c);
		so_fill_with_closest(tb1.x, tb1.y, data, w, h, c);
		so_fill_with_closest(tb2.x, tb2.y, data, w, h, c);
		so_fill_with_closest(tb3.x, tb3.y, data, w, h, c);

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

		so_connected_set_add(&currentSet, &sp);
	}

	connected_set_t *dstConnectedSet = 0;
	for (connected_set_t **connectedSet = connectedSets; *connectedSet; connectedSet = &(*connectedSet)->next)
	{
		retry:
		if (so_connected_sets_intersect(&currentSet, *connectedSet))
		{
			if (!dstConnectedSet) // found a set that the edge is connected to -> add current edge to that set
			{
				so_connected_sets_in_place_merge(*connectedSet, &currentSet);
				dstConnectedSet = *connectedSet;
			}
			else // found another set that the edge is connected to -> merge those sets
			{
				so_connected_sets_in_place_merge(dstConnectedSet, *connectedSet);

				// remove current connectedSet from connectedSets
				connected_set_t *toDelete = *connectedSet;
				*connectedSet = (*connectedSet)->next;
				so_connected_set_free(toDelete);
				so_free(toDelete);
				if (*connectedSet)
					goto retry; // don't move to next since we already did that by deleting the current set
				else
					break;
			}
		}
	}
	if (!dstConnectedSet) // did not find a set that the edge is connected to -> make a new one
	{
		currentSet.next = *connectedSets;
		*connectedSets = so_alloc(connected_set_t, 1);
		**connectedSets = currentSet;
	}
	else
		so_connected_set_free(&currentSet);
}

static void so_connected_sets_free(connected_set_t **connectedSets)
{
	connected_set_t *connectedSet = *connectedSets;
	while (connectedSet)
	{
		connected_set_t *next = connectedSet->next;
		so_connected_set_free(connectedSet);
		so_free(connectedSet);
		connectedSet = next;
	}
}

static int so_should_optimize(so_vec3 *tria, so_vec3 *trib)
{
	so_vec3 n0 = so_normalize3(so_cross3(so_sub3(tria[1], tria[0]), so_sub3(tria[2], tria[0])));
	so_vec3 n1 = so_normalize3(so_cross3(so_sub3(trib[1], trib[0]), so_sub3(trib[2], trib[0])));
	return so_absf(so_dot3(n0, n1)) > 0.99f; // TODO: make threshold an argument!
}

static int so_texel_binary_search(texel_t *texels, int n, texel_t toFind)
{
	int n_half = n / 2;
	texel_t *center = texels + n_half;
	if (toFind.y == center->y && toFind.x == center->x)
		return n_half;
	if (n <= 1)
		return -1;
	if (toFind.y < center->y || (toFind.y == center->y && toFind.x < center->x))
		return so_texel_binary_search(texels, n_half, toFind);
	else
	{
		int result = so_texel_binary_search(center + 1, n - n_half - 1, toFind);
		return result == -1 ? -1 : n_half + 1 + result;
	}
}

static void so_matrix_At_times_A(const float *A, int m, int n, float *AtA, const int *sparseIndices, int maxRowIndices)
{
	//if (!AtAisZero)
	//	memset(AtA, 0, n * n * sizeof(float));

	// compute lower left triangle only since the result is symmetric
	for (int k = 0; k < m; k++)
	{
		const float *srcPtr = A + k * maxRowIndices;
		const int *indexPtr = sparseIndices + k * maxRowIndices;
		for (int i = 0; i < maxRowIndices; i++)
		{
			int index_i = indexPtr[i];
			if (index_i < 0) break;
			float v = srcPtr[i];
			float *dstPtr = AtA + index_i * n;
			for (int j = 0; j < maxRowIndices; j++)
			{
				int index_j = indexPtr[j];
				if (index_j < 0) break;
				dstPtr[index_j] += v * srcPtr[j];
			}
		}
	}

	// mirror lower left triangle to upper right
	//for (int i = 0; i < n; i++)
	//	for (int j = 0; j < i; j++)
	//		AtA[j * n + i] = AtA[i * n + j];
}

static void so_matrix_At_times_b(const float *A, int m, int n, const float *b, float *Atb, const int *sparseIndices, int maxRowIndices)
{
	memset(Atb, 0, sizeof(float) * n);
	for (int j = 0; j < m; j++)
	{
		const int *rowIndices = sparseIndices + j * maxRowIndices;
		for (int i = 0; i < maxRowIndices; i++)
		{
			int index = rowIndices[i];
			if (index < 0) break;
			Atb[index] += A[j * maxRowIndices + i] * b[j];
		}
	}
}

static bool so_matrix_cholesky_prepare(const float *A, int n, float *L, int *sparseIndicesOut = 0)
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
				if (sum <= 0.0f) return false;
				invDiag[i] = so_rsqrtf(sum);
			}

			if (SO_NOT_ZERO(sum))
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

static void so_matrix_cholesky_solve(const float *L, float *x, const float *b, int n, const int *LsparseIndices = 0)
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
	float lambda;
	int w, h, c;
	float *data;
} so_seams_t;


static void so_optimize_seam(so_seams_t *data, connected_set_t *connectedSet)
{
	texel_set_t *texels = &connectedSet->texels;
	stitching_points_t *stitchingPoints = &connectedSet->stitchingPoints;

	size_t m = stitchingPoints->count;
	size_t n = texels->count;

	void *memoryBlock = so_alloc_void(
		sizeof(texel_t) * n +
		sizeof(float) * (m + n) * 8 +
		sizeof(int) * (m + n) * 8 +
		sizeof(float) * n * n +
		sizeof(float) * n * n + 
		sizeof(int) * n * n +
		sizeof(float) * (m + n) +
		sizeof(float) * n +
		sizeof(float) * n);

	uint8_t *memoryStart = (uint8_t*)memoryBlock;

	texel_t *texelsFlat = (texel_t*)memoryStart;
	memoryStart += sizeof(texel_t) * n;

	float *A = (float*)memoryStart;
	memoryStart += sizeof(float) * (m + n) * 8;

	int *AsparseIndices = (int*)memoryStart;
	memoryStart += sizeof(int) * (m + n) * 8;


	float *AtA = (float*)memoryStart;
	memoryStart += sizeof(float) * n * n;


	float *L = (float*)memoryStart;
	memoryStart += sizeof(float) * n * n;


	int *LsparseIndices = (int*)memoryStart;
	memoryStart += sizeof(int) * n * n;


	float *b = (float*)memoryStart;
	memoryStart += sizeof(float) * (m + n);


	float *Atb = (float*)memoryStart;
	memoryStart += sizeof(float) * n;


	float *x = (float*)memoryStart;
	memoryStart += sizeof(float) * n;

	/*texel_t *texelsFlat = so_alloc(texel_t, n);
	float *A = so_alloc(float, (m + n) * 8);
	int *AsparseIndices = so_alloc(int, (m + n) * 8);
	float *AtA = so_alloc(float, n * n);
	float *L = so_alloc(float, n * n);
	int *LsparseIndices = so_alloc(int, n * n);
	float *b = so_alloc(float, m + n);
	float *Atb = so_alloc(float, n);
	float *x = so_alloc(float, n);*/

	for (int i = 0, j = 0; i < texels->capacity && j < n; i++)
		if (texels->texels[i].x != -1)
			texelsFlat[j++] = texels->texels[i];

	qsort(texelsFlat, n, sizeof(texel_t), so_texel_cmp);

	size_t r = 0;
	for (int i = 0; i < m; i++)
	{
		ptrdiff_t column0[4];
		ptrdiff_t column1[4];
		bool side0valid = false, side1valid = false;
		for (int k = 0; k < 4; k++)
		{
			texel_t t0 = stitchingPoints->points[i].sides[0].texels[k];
			texel_t t1 = stitchingPoints->points[i].sides[1].texels[k];
			column0[k] = so_texel_binary_search(texelsFlat, n, t0);
			column1[k] = so_texel_binary_search(texelsFlat, n, t1);

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
				A[r * 8 + k * 2 + 0] = stitchingPoints->points[i].sides[0].weights[k];
				AsparseIndices[r * 8 + k * 2 + 0] = column0[k];
				A[r * 8 + k * 2 + 1] = -stitchingPoints->points[i].sides[1].weights[k];
				AsparseIndices[r * 8 + k * 2 + 1] = column1[k];
			}
			r++;
		}
	}

	m = r;

	// add error terms for deviation from original pixel value (scaled by lambda)
	for (int i = 0; i < n; i++)
	{
		A[(m + i) * 8] = data->lambda;
		AsparseIndices[(m + i) * 8 + 0] = i;
		AsparseIndices[(m + i) * 8 + 1] = -1;
	}

	so_matrix_At_times_A(A, m + n, n, AtA, AsparseIndices, 8);

	if (!so_matrix_cholesky_prepare(AtA, n, L, LsparseIndices))
	{
		printf("Could not compute cholesky decomposition.\n");
		so_free(memoryBlock);
		/*so_free(texelsFlat);
		so_free(A);
		so_free(AsparseIndices);
		so_free(AtA);
		so_free(L);
		so_free(LsparseIndices);*/
		return;
	}

	// solve each color channel independently
  	for (int ci = 0; ci < data->c; ci++)
	{
		for (int i = 0; i < n; i++)
			b[m + i] = data->lambda * data->data[(texelsFlat[i].y * data->w + texelsFlat[i].x) * data->c + ci];

		so_matrix_At_times_b(A, m + n, n, b, Atb, AsparseIndices, 8);
		so_matrix_cholesky_solve(L, x, Atb, n, LsparseIndices);

		// write out results
		for (int i = 0; i < n; i++)
			data->data[(texelsFlat[i].y * data->w + texelsFlat[i].x) * data->c + ci] = x[i];
	}

	so_free(memoryBlock);
	/*so_free(x);
	so_free(Atb);
	so_free(b);
	so_free(L);
	so_free(LsparseIndices);
	so_free(AtA);
	so_free(A);
	so_free(AsparseIndices);
	so_free(texelsFlat);*/
}

static void so_optimize_seams(float *positions, float *texcoords, int vertices, float *data, int w, int h, int c, float lambda)
{
	so_vec3 *pos = (so_vec3*)positions;
	so_vec2 *uv = (so_vec2*)texcoords;

	so_vec3 bbmin = so_v3(FLT_MAX, FLT_MAX, FLT_MAX);
	so_vec3 bbmax = so_v3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	int *hashmap = so_alloc(int, vertices * 2);
	for (int i = 0; i < vertices; i++)
	{
		bbmin = so_min3(bbmin, pos[i]);
		bbmax = so_max3(bbmax, pos[i]);
		hashmap[i * 2 + 0] = -1;
		hashmap[i * 2 + 1] = -1;
	}
	
	so_vec3 bbscale = so_v3(15.9f / bbmax.x, 15.9f / bbmax.y, 15.9f / bbmax.z);

	connected_set_t *connectedSets = 0;

	printf("Searching for sets of stitching points...\n");

	for (int i0 = 0; i0 < vertices; i0++)
	{
		int tri = i0 - (i0 % 3);
		int i1 = tri + ((i0 + 1) % 3);
		int i2 = tri + ((i0 + 2) % 3);
		so_vec3 p = so_mul3(so_sub3(pos[i0], bbmin), bbscale);
		int hash = (281 * (int)p.x + 569 * (int)p.y + 1447 * (int)p.z) % (vertices * 2);
		while (hashmap[hash] >= 0)
		{
			int oi0 = hashmap[hash];
#define SO_EQUAL(a, b) so_length3sq(so_sub3(pos[a], pos[b])) < 0.0000001f
			if (SO_EQUAL(oi0, i0))
			{
				int otri = oi0 - (oi0 % 3);
				int oi1 = otri + ((oi0 + 1) % 3);
				int oi2 = otri + ((oi0 + 2) % 3);
				if (SO_EQUAL(oi1, i1) && so_should_optimize(pos + tri, pos + otri))
					so_connected_sets_add_seam(&connectedSets, uv[i0], uv[i1], uv[oi0], uv[oi1], data, w, h, c);
				//else if (SO_EQUAL(oi1, i2) && so_should_optimize(pos + tri, pos + otri)) // this will already be detected by the other side of the seam!
				//	so_connected_sets_add_seam(&connectedSets, uv[i0], uv[i2], uv[oi0], uv[oi1], data, w, h, c);
				else if (SO_EQUAL(oi2, i1) && so_should_optimize(pos + tri, pos + otri))
					so_connected_sets_add_seam(&connectedSets, uv[i0], uv[i1], uv[oi0], uv[oi2], data, w, h, c);
			}
			if (++hash == vertices * 2)
				hash = 0;
		}
		hashmap[hash] = i0;
	}

	so_free(hashmap);

	int setCount = 0;
	for (connected_set_t *connectedSet = connectedSets; connectedSet; connectedSet = connectedSet->next)
		setCount++;

	printf("Solving %d sets of stitching points...\n", setCount);

	so_seams_t seams;
	seams.lambda = lambda;
	seams.w = w;
	seams.h = h;
	seams.c = c;
	seams.data = data;

#if 0 //def SO_CPP_THREADS
	std::thread *threads = new std::thread[connectedSets.size()];

#define SO_THREAD_THRESHOLD 512

	for (int setIndex = 0; setIndex < connectedSets.size(); setIndex++)
		if (connectedSets[setIndex].stitchingPoints.size() > SO_THREAD_THRESHOLD)
			threads[setIndex] = std::thread(so_optimize_seam, &seams, setIndex);

	for (int setIndex = 0; setIndex < connectedSets.size(); setIndex++)
		if (connectedSets[setIndex].stitchingPoints.size() <= SO_THREAD_THRESHOLD)
			so_optimize_seam(&seams, setIndex);

	for (int setIndex = 0; setIndex < connectedSets.size(); setIndex++)
		if (connectedSets[setIndex].stitchingPoints.size() > SO_THREAD_THRESHOLD)
			threads[setIndex].join();

	delete[] threads;
#else
	for (connected_set_t *connectedSet = connectedSets; connectedSet; connectedSet = connectedSet->next)
		so_optimize_seam(&seams, connectedSet);
#endif

	so_connected_sets_free(&connectedSets);

#ifdef SO_CHECK_FOR_MEMORY_LEAKS
	assert(so_allocated == 0);
#endif
}