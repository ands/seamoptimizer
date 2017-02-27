# seamoptimizer
A C/C++ single-file library that minimizes the hard transition errors of disjoint edges in lightmaps.
It is based on a idea presented by Michał Iwanicki in the talk ![Lighting Technology of "The Last Of Us"](http://miciwan.com/SIGGRAPH2013/Lighting%20Technology%20of%20The%20Last%20Of%20Us.pdf).
A least squares solver is used to find a minimal error solution to the problem of sampling along the edges between triangles that are mapped with disjoint lightmap regions.
This can improve the visual appearance at these discontinuities or "seams".


To paste the implementation into your project, insert the following lines:
```
#define SEAMOPTIMIZER_IMPLEMENTATION
#include "seamoptimizer.h"
```

Before optimizing a very bad UV mapping (each triangle edge is a seam):
![Sean Optimizer Before](https://github.com/ands/seamoptimizer/raw/master/example_images/city_triangles_not_optimized.png)
After optimizing the seams of the bad UV mapping:
![Sean Optimizer After](https://github.com/ands/seamoptimizer/raw/master/example_images/city_triangles_optimized.png)

# Example Usage
The following example finds and optimizes all the seams for some mesh geometry on a lightmap.
```
// only optimize seams between triangles that are on the same plane
// (where dot(A.normal, B.normal) > cosNormalThreshold):
const float cosNormalThreshold = 0.99f;

// how "important" the original color values are:
const float lambda = 0.1f;


printf("Searching for separate seams...\n");
so_seam_t *seams = so_seams_find(
	(float*)mesh->positions, (float*)mesh->texcoords, mesh->vertexCount,
	cosNormalThreshold,
	lightmap->data, lightmap->width, lightmap->height, lightmap->channelCount);


printf("Optimizing seams...\n");
for (so_seam_t *seam = seams; seam; seam = so_seam_next(seam))
{
	// NOTE: seams can also be optimized in parallel on separate threads!
	if (!so_seam_optimize(seam, lightmap->data, lightmap->width, lightmap->height, lightmap->channelCount, lambda))
		printf("Could not optimize a seam (Cholesky decomposition failed).\n");
}

printf("Done!\n");
so_seams_free(seams);
```

# Thanks
- To Michał Iwanicki for helping me with the transformation of the problem into a problem that can be solved by a least-squares solver
- To Dominik Lazarek for pointing me to Michał's presentation