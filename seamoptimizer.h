/***********************************************************
* A single header file lightmap seam optimization library  *
* https://github.com/ands/seamoptimizer                    *
* no warranty implied | use at your own risk               *
* author: Andreas Mantler (ands) | last change: 05.03.2017 *
*                                                          *
* License:                                                 *
* This software is in the public domain.                   *
* Where that dedication is not recognized,                 *
* you are granted a perpetual, irrevocable license to copy *
* and modify this file however you want.                   *
***********************************************************/

#ifndef SEAMOPTIMIZER_H
#define SEAMOPTIMIZER_H

#ifndef SO_CALLOC
#include <malloc.h> // calloc, free, alloca
#define SO_CALLOC(count, size) calloc(count, size)
#define SO_FREE(ptr) free(ptr)
#endif

typedef int so_bool;
#define SO_FALSE 0
#define SO_TRUE  1

typedef struct so_seam_t so_seam_t;

// API

// so_seams_find:
// Find all seams according to the specified triangulated geometry and its texture coordinates.
// This searches for edges that are shared by triangles, but are disjoint in UV space.

// positions: triangle array 3d positions ((x0, y0, z0), (x1, y1, z1), (x2, y2, z2)), ((x0, y0, z0), (x1, y1, z1), (x2, y2, z2)), ...
// texcoords: triangle array 2d uv coords (    (u0, v0),     (u1, v1),     (u2, v2)), (    (u0, v0),     (u1, v1),     (u2, v2)), ...
// vertices: total number of vertices ( = triangles * 3)

// cosNormalThreshold controls at which angles between neighbour triangles a seam should be considered.
// if dot(triangle A normal, triangle B normal) > cosNormalThreshold then the seam is included into the returned set.

// data, w, h, c specifies the lightmap data (data should be a w * h * c array of floats).
// w = lightmap width, h = lightmap height, c = number of lightmap channels (1..4).

// returns a linked list of the found seams.

// Warning: The data may be modified to fill empty (zeroed) edge texels with one of their closest neighbours if they are empty!
so_seam_t *so_seams_find(
	float *positions, float *texcoords, int vertices,
	float cosNormalThreshold,
	float *data, int w, int h, int c);


// so_seam_optimize:
// Optimize a single seam. Seams can be optimized in parallel on different threads.
// lambda: Weight that controls the deviation from the original color values (must be > 0).
//         Higher values => Less deviation from the original edge colors => more obvious seams.
//         Too low values => Optimizer may just choose black as the perfect color for all seam pixels.
// returns whether the optimization was successful.
so_bool so_seam_optimize(
	so_seam_t *seam,
	float *data, int w, int h, int c,
	float lambda);

// so_seam_next: Retrieves the next seam in the linked list.
so_seam_t *so_seam_next(
	so_seam_t *seam);

// so_seams_free: Free the resources for all seams in the list.
void so_seams_free(
	so_seam_t *seams);

#endif
////////////////////// END OF HEADER //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef SEAMOPTIMIZER_IMPLEMENTATION
#undef SEAMOPTIMIZER_IMPLEMENTATION

#include <stdlib.h> // qsort
#include <stdio.h> // printf (TODO)
#include <string.h> // memcpy
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <assert.h>

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
static uint64_t so_allocated_max = 0;

static void *so_alloc_void(size_t size)
{
	void *memory = SO_CALLOC(1, size + sizeof(size_t));
	(*(size_t*)memory) = size;
	so_allocated += size;
	if (so_allocated > so_allocated_max)
		so_allocated_max = so_allocated;
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
	return SO_CALLOC(1, size);
}
static void so_free(void *memory)
{
	SO_FREE(memory);
}
#endif

#define so_alloc(type, count) ((type*)so_alloc_void(sizeof(type) * (count)))

static inline so_bool so_accumulate_texel(float *sums, int x, int y, float *data, int w, int h, int c)
{
	so_bool exists = SO_FALSE;
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

typedef struct
{
	int16_t x, y;
} so_texel_t;

static inline int so_texel_cmp(const void *l, const void *r)
{
	const so_texel_t *lt = (const so_texel_t*)l;
	const so_texel_t *rt = (const so_texel_t*)r;
	if (lt->y < rt->y) return -1;
	if (lt->y > rt->y) return 1;
	if (lt->x < rt->x) return -1;
	if (lt->x > rt->x) return 1;
	return 0;
}

typedef struct
{
	so_texel_t texels[4];
	float weights[4];
} so_bilinear_sample_t;

typedef struct
{
	so_bilinear_sample_t sides[2];
} so_stitching_point_t;

typedef struct 
{
	so_texel_t *texels;
	uint32_t count;
	uint32_t capacity;
} so_texel_set_t;

static inline uint32_t so_texel_hash(so_texel_t texel, uint32_t capacity)
{
	return (texel.y * 104173 + texel.x * 86813) % capacity;
}

static void so_texel_set_add(so_texel_set_t *set, so_texel_t *texels, int entries, int arrayLength = 0)
{
	if (set->count + entries > set->capacity * 3 / 4) // leave some free space to avoid having many collisions
	{
		int newCapacity = set->capacity > 64 ? set->capacity * 2 : 64;
		while (set->count + entries > newCapacity * 3 / 4)
			newCapacity *= 2;

		so_texel_t *newTexels = so_alloc(so_texel_t, newCapacity);

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

static so_bool so_texel_set_contains(so_texel_set_t *set, so_texel_t texel)
{
	uint32_t hash = so_texel_hash(texel, set->capacity);
	while (set->texels[hash].x != -1) // entries with same hash
	{
		if (set->texels[hash].x == texel.x && set->texels[hash].y == texel.y)
			return SO_TRUE; // texel is already in the set
		hash = (hash + 1) % set->capacity;
	}
	return SO_FALSE;
}

static void so_texel_set_free(so_texel_set_t *set)
{
	so_free(set->texels);
	*set = {0};
}

typedef struct 
{
	so_stitching_point_t *points;
	uint32_t count;
	uint32_t capacity;
} so_stitching_points_t;

static void so_stitching_points_alloc(so_stitching_points_t *points, uint32_t n)
{
	points->points = so_alloc(so_stitching_point_t, n);
	points->capacity = n;
	points->count = 0;
}
static void so_stitching_points_free(so_stitching_points_t *points)
{
	so_free(points->points);
	*points = {0};
}
static void so_stitching_points_add(so_stitching_points_t *points, so_stitching_point_t *point)
{
	assert(points->count < points->capacity);
	points->points[points->count++] = *point;
}
static void so_stitching_points_append(so_stitching_points_t *points, so_stitching_points_t *other)
{
	so_stitching_point_t *newPoints = so_alloc(so_stitching_point_t, points->capacity + other->capacity);
	memcpy(newPoints, points->points, sizeof(so_stitching_point_t) * points->count);
	memcpy(newPoints + points->count, other->points, sizeof(so_stitching_point_t) * other->count);
	so_free(points->points);
	points->points = newPoints;
	points->capacity = points->capacity + other->capacity;
	points->count = points->count + other->count;
}

struct so_seam_t
{
	int16_t x_min, y_min, x_max, y_max;
	so_texel_set_t texels;
	so_stitching_points_t stitchingPoints;
	so_seam_t *next;
};

so_seam_t *so_seam_next(so_seam_t *seam)
{
	return seam->next;
}

static void so_seam_alloc(so_seam_t *seam, uint32_t stitchingPointCount)
{
	so_stitching_points_alloc(&seam->stitchingPoints, stitchingPointCount);
}
static void so_seam_free(so_seam_t *seam)
{
	so_texel_set_free(&seam->texels);
	so_stitching_points_free(&seam->stitchingPoints);
}

static void so_seam_add(so_seam_t *seam, so_stitching_point_t *point)
{
	for (int side = 0; side < 2; side++)
	{
		for (int texel = 0; texel < 4; texel++)
		{
			so_texel_t t = point->sides[side].texels[texel];
			seam->x_min = t.x < seam->x_min ? t.x : seam->x_min;
			seam->y_min = t.y < seam->y_min ? t.y : seam->y_min;
			seam->x_max = t.x > seam->x_max ? t.x : seam->x_max;
			seam->y_max = t.y > seam->y_max ? t.y : seam->y_max;
		}
		so_texel_set_add(&seam->texels, point->sides[side].texels, 4);
	}

	so_stitching_points_add(&seam->stitchingPoints, point);
}

static so_bool so_seams_intersect(so_seam_t *a, so_seam_t *b)
{
	// compare bounding boxes first
	if (a->x_min > b->x_max || b->x_min >= a->x_max ||
		a->y_min > b->y_max || b->y_min >= a->y_max)
		return SO_FALSE;

	// bounds intersect -> check each individual texel for intersection
	if (a->texels.capacity > b->texels.capacity) // swap so that we always loop over the smaller set
	{
		so_seam_t *tmp = a;
		a = b;
		b = tmp;
	}

	for (int i = 0; i < a->texels.capacity; i++)
		if (a->texels.texels[i].x != -1)
			if (so_texel_set_contains(&b->texels, a->texels.texels[i]))
				return SO_TRUE;
	return SO_FALSE;
}

static void so_seams_in_place_merge(so_seam_t *dst, so_seam_t *src)
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

static void so_seams_add_seam(so_seam_t **seams, so_vec2 a0, so_vec2 a1, so_vec2 b0, so_vec2 b1, float *data, int w, int h, int c)
{
	so_vec2 s = so_v2i(w, h);
	a0 = so_mul2(a0, s);
	a1 = so_mul2(a1, s);
	b0 = so_mul2(b0, s);
	b1 = so_mul2(b1, s);
	so_vec2 ad = so_sub2(a1, a0);
	so_vec2 bd = so_sub2(b1, b0);
	float l = so_length2(ad);
	int iterations = (int)(l * 5.0f); // TODO: is this the best value?
	float step = 1.0f / iterations;

	so_seam_t currentSeam = {0};
	currentSeam.x_min = w; currentSeam.y_min = h;
	currentSeam.x_max = 0; currentSeam.y_max = 0;

	so_seam_alloc(&currentSeam, iterations + 1);

	for (int i = 0; i <= iterations; i++)
	{
		float t = i * step;
		so_vec2 a = so_add2(a0, so_scale2(ad, t));
		so_vec2 b = so_add2(b0, so_scale2(bd, t));
		int16_t ax = (int16_t)roundf(a.x), ay = (int16_t)roundf(a.y);
		int16_t bx = (int16_t)roundf(b.x), by = (int16_t)roundf(b.y);
		float au = a.x - ax, av = a.y - ay, nau = 1.0f - au, nav = 1.0f - av;
		float bu = b.x - bx, bv = b.y - by, nbu = 1.0f - bu, nbv = 1.0f - bv;

		so_texel_t ta0 = { ax    , ay     };
		so_texel_t ta1 = { ax + 1, ay     };
		so_texel_t ta2 = { ax    , ay + 1 };
		so_texel_t ta3 = { ax + 1, ay + 1 };

		so_texel_t tb0 = { bx    , by     };
		so_texel_t tb1 = { bx + 1, by     };
		so_texel_t tb2 = { bx    , by + 1 };
		so_texel_t tb3 = { bx + 1, by + 1 };

		so_fill_with_closest(ta0.x, ta0.y, data, w, h, c);
		so_fill_with_closest(ta1.x, ta1.y, data, w, h, c);
		so_fill_with_closest(ta2.x, ta2.y, data, w, h, c);
		so_fill_with_closest(ta3.x, ta3.y, data, w, h, c);

		so_fill_with_closest(tb0.x, tb0.y, data, w, h, c);
		so_fill_with_closest(tb1.x, tb1.y, data, w, h, c);
		so_fill_with_closest(tb2.x, tb2.y, data, w, h, c);
		so_fill_with_closest(tb3.x, tb3.y, data, w, h, c);

		so_stitching_point_t sp;
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

		so_seam_add(&currentSeam, &sp);
	}

	so_seam_t *dstSeam = 0;
	for (so_seam_t **seam = seams; *seam; seam = &(*seam)->next)
	{
		retry:
		if (so_seams_intersect(&currentSeam, *seam))
		{
			if (!dstSeam) // found a seam that the edge is connected to -> add current edge to that seam
			{
				so_seams_in_place_merge(*seam, &currentSeam);
				dstSeam = *seam;
			}
			else // found another seam that the edge is connected to -> merge those seams
			{
				so_seams_in_place_merge(dstSeam, *seam);

				// remove current seam from seams
				so_seam_t *toDelete = *seam;
				*seam = (*seam)->next;
				so_seam_free(toDelete);
				so_free(toDelete);
				if (*seam)
					goto retry; // don't move to next since we already did that by deleting the current seam
				else
					break;
			}
		}
	}
	if (!dstSeam) // did not find a seam that the edge is connected to -> make a new one
	{
		currentSeam.next = *seams;
		*seams = so_alloc(so_seam_t, 1);
		**seams = currentSeam;
	}
	else
		so_seam_free(&currentSeam);
}

void so_seams_free(so_seam_t *seams)
{
	so_seam_t *seam = seams;
	while (seam)
	{
		so_seam_t *next = seam->next;
		so_seam_free(seam);
		so_free(seam);
		seam = next;
	}

#ifdef SO_CHECK_FOR_MEMORY_LEAKS
	assert(so_allocated == 0);
	printf("Allocated max %d MB. Not freed: %d bytes.\n", so_allocated_max / (1024 * 1024), so_allocated);
	printf("These results are only correct if the lib was used single-threaded.\n");
	so_allocated_max = 0;
#endif
}

static int so_should_optimize(so_vec3 *tria, so_vec3 *trib, float cosThreshold)
{
	so_vec3 n0 = so_normalize3(so_cross3(so_sub3(tria[1], tria[0]), so_sub3(tria[2], tria[0])));
	so_vec3 n1 = so_normalize3(so_cross3(so_sub3(trib[1], trib[0]), so_sub3(trib[2], trib[0])));
	return so_absf(so_dot3(n0, n1)) > cosThreshold;
}

so_seam_t *so_seams_find(float *positions, float *texcoords, int vertices, float cosNormalThreshold, float *data, int w, int h, int c)
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

	so_seam_t *seams = 0;

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
				if (SO_EQUAL(oi1, i1) && so_should_optimize(pos + tri, pos + otri, cosNormalThreshold))
					so_seams_add_seam(&seams, uv[i0], uv[i1], uv[oi0], uv[oi1], data, w, h, c);
				//else if (SO_EQUAL(oi1, i2) && so_should_optimize(pos + tri, pos + otri, cosNormalThreshold)) // this will already be detected by the other side of the seam!
				//	so_seams_add_seam(&seams, uv[i0], uv[i2], uv[oi0], uv[oi1], data, w, h, c);
				else if (SO_EQUAL(oi2, i1) && so_should_optimize(pos + tri, pos + otri, cosNormalThreshold))
					so_seams_add_seam(&seams, uv[i0], uv[i1], uv[oi0], uv[oi2], data, w, h, c);
				//break;
			}
			if (++hash == vertices * 2)
				hash = 0;
		}
		hashmap[hash] = i0;
	}

	so_free(hashmap);
	return seams;
}

static int so_texel_binary_search(so_texel_t *texels, int n, so_texel_t toFind)
{
	int n_half = n / 2;
	so_texel_t *center = texels + n_half;
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

typedef struct
{
	int index;
	float value;
} so_sparse_entry_t;

static int so_sparse_entry_cmp(const void *a, const void *b)
{
	so_sparse_entry_t *ae = (so_sparse_entry_t*)a;
	so_sparse_entry_t *be = (so_sparse_entry_t*)b;
	return ae->index - be->index;
}

typedef struct
{
	so_sparse_entry_t *entries;
	int count;
	int capacity;
} so_sparse_entries_t;

static void so_sparse_matrix_alloc(so_sparse_entries_t *matrix, int capacity)
{
	matrix->entries = so_alloc(so_sparse_entry_t, capacity);
	matrix->capacity = capacity;
	matrix->count = 0;
}

static void so_sparse_matrix_free(so_sparse_entries_t *matrix)
{
	so_free(matrix->entries);
	*matrix = { 0 };
}

static void so_sparse_matrix_add(so_sparse_entries_t *matrix, int index, float value)
{
	if (matrix->count == matrix->capacity)
	{
		int newCapacity = matrix->capacity * 2;
		if (newCapacity < 64)
			newCapacity = 64;
		so_sparse_entry_t *newEntries = so_alloc(so_sparse_entry_t, newCapacity);
		for (int i = 0; i < matrix->count; i++)
			newEntries[i] = matrix->entries[i];
		so_free(matrix->entries);
		matrix->entries = newEntries;
		matrix->capacity = newCapacity;
	}

	int entryIndex = matrix->count++;
	matrix->entries[entryIndex].index = index;
	matrix->entries[entryIndex].value = value;
}

static void so_sparse_matrix_add(so_sparse_entries_t *matrix, so_sparse_entry_t *entry)
{
	so_sparse_matrix_add(matrix, entry->index, entry->value);
}

static void so_sparse_matrix_sort(so_sparse_entries_t *matrix)
{
	qsort(matrix->entries, matrix->count, sizeof(so_sparse_entry_t), so_sparse_entry_cmp);
}

static so_bool so_sparse_matrix_advance_to_index(so_sparse_entries_t *matrix, int *position, int index, float *outValue)
{
	int localPosition = *position;
	while (localPosition < matrix->count && matrix->entries[localPosition].index < index)
		++localPosition;
	*position = localPosition;

	if (localPosition < matrix->count && matrix->entries[localPosition].index == index)
	{
		*outValue = matrix->entries[localPosition].value;
		return SO_TRUE;
	}

	return SO_FALSE;
}

static inline uint32_t so_sparse_entry_hash(int entryIndex, uint32_t capacity)
{
	return (entryIndex * 104173) % capacity;
}

static void so_sparse_entry_set_alloc(so_sparse_entries_t *set, int capacity)
{
	set->entries = so_alloc(so_sparse_entry_t, capacity);
	for (int i = 0; i < capacity; i++)
		set->entries[i].index = -1;
	set->capacity = capacity;
	set->count = 0;
}

static so_sparse_entry_t *so_sparse_entry_set_get_or_add(so_sparse_entries_t *set, int index)
{
	if (set->count + 1 > set->capacity * 3 / 4) // leave some free space to avoid having many collisions
	{
		int newCapacity = set->capacity >= 64 ? set->capacity * 2 : 64;
		so_sparse_entry_t *newEntries = so_alloc(so_sparse_entry_t, newCapacity);
		for (int i = 0; i < newCapacity; i++)
			newEntries[i].index = -1;

		for (int i = 0; i < set->capacity; i++) // rehash all old entries
		{
			if (set->entries[i].index != -1)
			{
				uint32_t hash = so_sparse_entry_hash(set->entries[i].index, newCapacity);
				while (newEntries[hash].index != -1) // collisions
					hash = (hash + 1) % newCapacity;
				newEntries[hash] = set->entries[i];
			}
		}
		so_free(set->entries);
		set->entries = newEntries;
		set->capacity = newCapacity;
	}

	uint32_t hash = so_sparse_entry_hash(index, set->capacity);
	while (set->entries[hash].index != -1) // collisions
	{
		if (set->entries[hash].index == index)
			return &set->entries[hash]; // entry is already in the set
		hash = (hash + 1) % set->capacity;
	}

	if (set->entries[hash].index == -1) // make new entry
	{
		set->entries[hash].index = index;
		set->entries[hash].value = 0.0f;
		set->count++;
		return &set->entries[hash];
	}

	return 0; // shouldn't happen
}

static so_sparse_entries_t so_matrix_At_times_A(const float *A, const int *sparseIndices, int maxRowIndices, int m, int n)
{
	so_sparse_entries_t AtA;
	so_sparse_entry_set_alloc(&AtA, (n / 16) * (n / 16));

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
			//float *dstPtr = AtA + index_i * n;
			for (int j = 0; j < maxRowIndices; j++)
			{
				int index_j = indexPtr[j];
				if (index_j < 0) break;
				//dstPtr[index_j] += v * srcPtr[j];
				int index = index_i * n + index_j;

				so_sparse_entry_t *entry = so_sparse_entry_set_get_or_add(&AtA, index);
				entry->value += v * srcPtr[j];
			}
		}
	}

	// compaction step (make a compact array from the scattered hash set values)
	for (int i = 0, j = 0; i < AtA.capacity; i++)
		if (AtA.entries[i].index != -1)
			AtA.entries[j++] = AtA.entries[i];

	// sort by index -> this is a sparse matrix now
	so_sparse_matrix_sort(&AtA);

	return AtA;
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

static so_sparse_entries_t so_matrix_cholesky_prepare(so_sparse_entries_t *AtA, int n)
{
	// dense
	//for (int i = 0; i < n; i++)
	//{
	//	float *a = L + i * n;
	//	for (int j = 0; j <= i; j++)
	//	{
	//		float *b = L + j * n;
	//		float sum = A[i * n + j];// + (i == j ? 0.0001 : 0.0); // some regularization
	//		for (int k = 0; k < j; k++)
	//			sum -= a[k] * b[k];
	//		if (i > j)
	//			a[j] = sum / b[j];
	//		else // i == j
	//		{
	//			if (sum <= 0.0)
	//				return SO_FALSE;
	//			a[i] = sqrtf(sum);
	//		}
	//	}
	//}
	
	// sparse
	int *indices_i = (int*)alloca(sizeof(int) * n);
	float *row_i = (float*)alloca(sizeof(float) * n);
	float *invDiag = (float*)alloca(sizeof(float) * n);

	so_sparse_entries_t L;
	so_sparse_matrix_alloc(&L, (n / 16) * (n / 16));

	int AtAindex = 0;
	for (int i = 0; i < n; i++)
	{
		int index_i_count = 0;

		int row_j_index = 0;
		for (int j = 0; j <= i; j++)
		{
			//float sum = A[i * n + j]; // + (i == j ? 0.0001 : 0.0); // regularization
			int index = i * n + j;
			float sum = 0.0f;
			so_sparse_matrix_advance_to_index(AtA, &AtAindex, index, &sum);

			for (int k = 0; k < index_i_count; k++)
			{
				int index_i = indices_i[k];
				float Lvalue;
				if (so_sparse_matrix_advance_to_index(&L, &row_j_index, j * n + index_i, &Lvalue))
					sum -= row_i[index_i] * Lvalue;
			}

			if (i == j)
			{
				if (sum <= 0.0f)
				{
					so_sparse_matrix_free(&L);
					return L;
				}
				invDiag[i] = so_rsqrtf(sum);
			}

			if (SO_NOT_ZERO(sum))
			{
				row_i[j] = sum * invDiag[j];
				indices_i[index_i_count++] = j;
				so_sparse_matrix_add(&L, index, row_i[j]);
			}
			else
				row_i[j] = 0.0f;
		}
	}

	return L;
}

static void so_matrix_cholesky_solve(so_sparse_entries_t *Lrows, so_sparse_entries_t *Lcols, float *x, const float *b, int n)
{
	float *y = (float*)alloca(sizeof(float) * n);

	// L * y = b
	int Lindex = 0;
	for (int i = 0; i < n; i++)
	{
		float sum = b[i];
		while (Lindex < Lrows->count && Lrows->entries[Lindex].index < i * (n + 1))
		{
			sum -= Lrows->entries[Lindex].value * y[Lrows->entries[Lindex].index - i * n];
			++Lindex;
		}
		assert(Lrows->entries[Lindex].index == i * (n + 1));
		y[i] = sum / Lrows->entries[Lindex].value;
		++Lindex;
	}

	// L' * x = y
	Lindex = Lcols->count - 1;
	for (int i = n - 1; i >= 0; i--)
	{
		float sum = y[i];
		while (Lindex >= 0 && Lcols->entries[Lindex].index > i * (n + 1))
		{
			sum -= Lcols->entries[Lindex].value * x[Lcols->entries[Lindex].index - i * n];
			--Lindex;
		}
		assert(Lcols->entries[Lindex].index == i * (n + 1));
		x[i] = sum / Lcols->entries[Lindex].value;
		--Lindex;
	}
}

so_bool so_seam_optimize(so_seam_t *seam, float *data, int w, int h, int c, float lambda)
{
	so_texel_set_t *texels = &seam->texels;
	so_stitching_points_t *stitchingPoints = &seam->stitchingPoints;

	size_t m = stitchingPoints->count;
	size_t n = texels->count;

	void *memoryBlock = so_alloc_void(
		sizeof(so_texel_t) * n +
		sizeof(float) * (m + n) * 8 +
		sizeof(int) * (m + n) * 8 +
		sizeof(float) * (m + n) +
		sizeof(float) * n +
		sizeof(float) * n);

	uint8_t *memoryStart = (uint8_t*)memoryBlock;

	so_texel_t *texelsFlat = (so_texel_t*)memoryStart;
	memoryStart += sizeof(so_texel_t) * n;

	float *A = (float*)memoryStart;
	memoryStart += sizeof(float) * (m + n) * 8;

	int *AsparseIndices = (int*)memoryStart;
	memoryStart += sizeof(int) * (m + n) * 8;

	float *b = (float*)memoryStart;
	memoryStart += sizeof(float) * (m + n);

	float *Atb = (float*)memoryStart;
	memoryStart += sizeof(float) * n;

	float *x = (float*)memoryStart;
	memoryStart += sizeof(float) * n;

	for (int i = 0, j = 0; i < texels->capacity && j < n; i++)
		if (texels->texels[i].x != -1)
			texelsFlat[j++] = texels->texels[i];

	qsort(texelsFlat, n, sizeof(so_texel_t), so_texel_cmp);

	size_t r = 0;
	for (int i = 0; i < m; i++)
	{
		ptrdiff_t column0[4];
		ptrdiff_t column1[4];
		so_bool side0valid = SO_FALSE, side1valid = SO_FALSE;
		for (int k = 0; k < 4; k++)
		{
			so_texel_t t0 = stitchingPoints->points[i].sides[0].texels[k];
			so_texel_t t1 = stitchingPoints->points[i].sides[1].texels[k];
			column0[k] = so_texel_binary_search(texelsFlat, n, t0);
			column1[k] = so_texel_binary_search(texelsFlat, n, t1);

			if (column0[k] == -1) { side0valid = SO_FALSE; break; }
			if (column1[k] == -1) { side1valid = SO_FALSE; break; }

			// test for validity of stitching point
			for (int ci = 0; ci < c; ci++)
			{
				side0valid |= data[(t0.y * w + t0.x) * c + ci] > 0.0f;
				side1valid |= data[(t1.y * w + t1.x) * c + ci] > 0.0f;
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
		A[(m + i) * 8] = lambda;
		AsparseIndices[(m + i) * 8 + 0] = i;
		AsparseIndices[(m + i) * 8 + 1] = -1;
	}

	so_sparse_entries_t AtA = so_matrix_At_times_A(A, AsparseIndices, 8, m + n, n);
	so_sparse_entries_t L = so_matrix_cholesky_prepare(&AtA, n);
	so_sparse_matrix_free(&AtA);

	if (!L.count)
	{
		so_free(memoryBlock);
		return SO_FALSE; // Cholesky decomposition failed
	}

	so_sparse_entries_t Lcols;
	so_sparse_matrix_alloc(&Lcols, L.count);
	for (int i = 0; i < L.count; i++)
		so_sparse_matrix_add(&Lcols, (L.entries[i].index % n) * n + (L.entries[i].index / n), L.entries[i].value);
	so_sparse_matrix_sort(&Lcols);

	// solve each color channel independently
  	for (int ci = 0; ci < c; ci++)
	{
		for (int i = 0; i < n; i++)
			b[m + i] = lambda * data[(texelsFlat[i].y * w + texelsFlat[i].x) * c + ci];

		so_matrix_At_times_b(A, m + n, n, b, Atb, AsparseIndices, 8);
		so_matrix_cholesky_solve(&L, &Lcols, x, Atb, n);

		// write out results
		for (int i = 0; i < n; i++)
			data[(texelsFlat[i].y * w + texelsFlat[i].x) * c + ci] = x[i];
	}

	so_free(memoryBlock);
	so_sparse_matrix_free(&L);
	so_sparse_matrix_free(&Lcols);

	return SO_TRUE;
}

#endif // SEAMOPTIMIZER_IMPLEMENTATION
