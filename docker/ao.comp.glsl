#version 450

layout(local_size_x = 192, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer inputs
{
  vec2 iResolution;
  vec2 offset;
  vec2 mouse;
  float iTime;
  float iFrame;
};

layout(std430, binding = 1) buffer outputs
{
  uint imageData[];
};

/*
  by kioku / System K
  https://www.shadertoy.com/view/ldl3Rn

  [ Modified to add normalized resolution and supersampling and attempt at a less bandy PRNG. ]
*/
/*
    AOBench normalized to 1920x1080 with 4x4 supersampling and 256 AO rays per primary.

    This is for comparing with CPUs using an ISPC build at similar settings.

    Results table:

    WebGL
    -----
    0.033s Nvidia RTX 2070
    0.1s Nvidia GTX 1050 4GB
    0.2s Intel Iris Graphics 655
    0.6s Intel UHD Graphics 630
    1.8s Snapdragon 835
    5.2s Snapdragon 820

    ISPC
    ----
    0.6s AMD TR 2950X [16c HT, ~AVX2]
    1.2s Dual-Xeon E5-2650 v2 [16c HT, AVX]
    1.5s i5-8259U [4c HT, AVX2]
    1.6s Xeon E3-1231v3 [4c HT, AVX2]
    2.9s i7-3770 [4c HT, AVX]
    7.8s i5-3317U [2c HT, AVX]
    11s Snapdragon 835 [8c, 4+4c, NEON, aarch64]
    15s Exynos 7420 Octa [8c, 4 a53 + 4 a57, NEON, aarch64]
    19s Snapdragon 820 [4c, 2+2c, NEON, aarch64]
    55s Raspberry PI 3B+ [4c a53, NEON, armv7l]
    56s AllWinner H5 [4c a53, NEON, aarch64]
    100s AllWinner H3 [4c a7, NEON, armv7l]

    FWIW, single-threaded scalar C++ on the fast x86s runs in ~50-60s.
    Extrapolating that, WASM ~75s, JS 150s+, Workers for ~4-40s runtimes [16c wasm / 4c JS]

*/

// To benchmark, uncomment this. For most representative results, run fullscreen at 1920x1080. Count the cyan bars for fps :P
// Frame times above are calculated as 1/fps.
//
#define BENCHMARK

// Control benchmark heaviness.
// Set this to 1.0 or even 0.5 for mobile SoCs.
// To normalize frame times:
//   2.0 subsamples - divide fps by 4
//   1.0 - divide fps by 16
//   0.5 - divide fps by 64
//
#define BENCHMARK_SUBSAMPLES 4.0

// Control demo mode supersampling (SUBSAMPLES * SUBSAMPLES primary rays per pixel)
#define SUBSAMPLES 2.0

mat3 rotationXY(vec2 angle)
{
  float cp = cos(angle.x);
  float sp = sin(angle.x);
  float cy = cos(angle.y);
  float sy = sin(angle.y);

  return mat3(
             cy, -sy, 0.0,
             sy, cy, 0.0,
             0.0, 0.0, 1.0) *
         mat3(
             cp, 0.0, -sp,
             0.0, 1.0, 0.0,
             sp, 0.0, cp);
}

struct Ray
{
  vec3 org;
  vec3 dir;
};
struct Sphere
{
  vec3 center;
  float radius;
};
struct Plane
{
  vec3 p;
  vec3 n;
};

struct Intersection
{
  float t;
  vec3 p; // hit point
  vec3 n; // normal
  int hit;
};

Sphere sphere[3];
Plane plane;
float aspectRatio = 16.0 / 9.0;

void shpere_intersect(Sphere s, Ray ray, inout Intersection isect)
{
  vec3 rs = ray.org - s.center;
  float B = dot(rs, ray.dir);
  float C = dot(rs, rs) - (s.radius * s.radius);
  float D = B * B - C;

  if (D > 0.0)
  {
    float t = -B - sqrt(D);
    if ((t > 0.0) && (t < isect.t))
    {
      isect.t = t;
      isect.hit = 1;

      // calculate normal.
      isect.p = ray.org + ray.dir * t;
      isect.n = normalize(isect.p - s.center);
    }
  }
}
void plane_intersect(Plane pl, Ray ray, inout Intersection isect)
{
  float d = -dot(pl.p, pl.n);
  float v = dot(ray.dir, pl.n);

  if (abs(v) < 1.0e-6)
  {
    return;
  }
  else
  {
    float t = -(dot(ray.org, pl.n) + d) / v;

    if ((t > 0.0) && (t < isect.t))
    {
      isect.hit = 1;
      isect.t = t;
      isect.n = pl.n;
      isect.p = ray.org + t * ray.dir;
    }
  }
}

void Intersect(Ray r, inout Intersection i)
{
  for (int c = 0; c < 3; c++)
  {
    shpere_intersect(sphere[c], r, i);
  }
  plane_intersect(plane, r, i);
}

void orthoBasis(out vec3 basis[3], vec3 n)
{
  basis[2] = vec3(n.x, n.y, n.z);
  basis[1] = vec3(0.0, 0.0, 0.0);

  if ((n.x < 0.6) && (n.x > -0.6))
    basis[1].x = 1.0;
  else if ((n.y < 0.6) && (n.y > -0.6))
    basis[1].y = 1.0;
  else if ((n.z < 0.6) && (n.z > -0.6))
    basis[1].z = 1.0;
  else
    basis[1].x = 1.0;

  basis[0] = cross(basis[1], basis[2]);
  basis[0] = normalize(basis[0]);

  basis[1] = cross(basis[2], basis[0]);
  basis[1] = normalize(basis[1]);
}

int seed = 0;

float randomn()
{
  seed = int(mod(float(seed) * 1364.0 + 626.0, 5209.0));
  return float(seed) / 5209.0;
}

float hash2(vec2 n)
{
  return fract(sin(dot(n, vec2(18.99221414, 15.839399))) * 13454.111388);
}

float computeAO(inout Intersection isect)
{
  const int ntheta = 8;
  const int nphi = 8;
  const float eps = 0.0001;

  // Slightly move ray org towards ray dir to avoid numerical problem.
  vec3 p = isect.p + eps * isect.n;

  // Calculate orthogonal basis.
  vec3 basis[3];
  orthoBasis(basis, isect.n);

  float occlusion = 0.0;

  for (int j = 0; j < ntheta; j++)
  {
    for (int i = 0; i < nphi; i++)
    {
      // Pick a random ray direction with importance sampling.
      // p = cos(theta) / 3.141592
      float r = randomn(); //hash2(isect.p.xy+vec2(i,j));
      float phi = 2.0 * 3.141592 * hash2(isect.p.xy + vec2(float(i) * 9.1, float(j) * 9.1));

      vec3 ref;
      float s, c;
      s = sin(phi);
      c = cos(phi);
      ref.x = c * sqrt(1.0 - r);
      ref.y = s * sqrt(1.0 - r);
      ref.z = sqrt(r);

      // local -> global
      vec3 rray;
      rray.x = ref.x * basis[0].x + ref.y * basis[1].x + ref.z * basis[2].x;
      rray.y = ref.x * basis[0].y + ref.y * basis[1].y + ref.z * basis[2].y;
      rray.z = ref.x * basis[0].z + ref.y * basis[1].z + ref.z * basis[2].z;

      vec3 raydir = vec3(rray.x, rray.y, rray.z);

      Ray ray;
      ray.org = p;
      ray.dir = raydir;

      Intersection occIsect;
      occIsect.hit = 0;
      occIsect.t = 1.0e30;
      occIsect.n = occIsect.p = vec3(0);
      Intersect(ray, occIsect);
      occlusion += (occIsect.hit != 0 ? 1.0 : 0.0);
    }
  }

  // [0.0, 1.0]
  occlusion = (float(ntheta * nphi) - occlusion) / float(ntheta * nphi);
  return occlusion;
}

void main()
{
  int width = int(iResolution.x);
  int height = int(iResolution.y);
  vec2 fragCoord = gl_GlobalInvocationID.xy + offset;
  fragCoord.y = iResolution.y - fragCoord.y;

  vec2 uv = fragCoord.xy / iResolution.xy;
  vec2 duv = ((fragCoord.xy + 1.0) / iResolution.xy) - uv;
  float fragColor = 0.0;
  seed = int(mod((fragCoord.x + 0.5) * (fragCoord.y * iResolution.y + 0.5), 65536.0));
#ifdef BENCHMARK
  float subSamples = BENCHMARK_SUBSAMPLES;
  for (float y = 0.; y < subSamples; y++)
  {
    for (float x = 0.; x < subSamples; x++)
    {
#else
  const float subSamples = SUBSAMPLES;
  for (float y = 0.; y < SUBSAMPLES; y++)
  {
    for (float x = 0.; x < SUBSAMPLES; x++)
    {
      float dt = iTimeDelta * (x + y * subSamples + randomn()) / (subSamples * subSamples + 1.0);
      float t = iTime - dt;
      mat3 rot = rotationXY(vec2(t, cos(t * 0.3) * 0.05));
#endif
      vec2 fuv = uv + (duv / subSamples * vec2(x, y));

      vec3 dir = normalize(vec3((fuv - 0.5) * 2.0 * vec2(1.0, 1.0 / aspectRatio), -1.0));
      Ray ray;
#ifdef BENCHMARK
      ray.org = vec3(0.0);
      ray.dir = dir;
#else
      ray.org = 3.5 * vec3(sin(t), 0, cos(t)) - vec3(0.0, 0.0, 3.0);
      ray.dir = normalize(rot * dir);
#endif
      Intersection it;
      it.hit = 0;
      it.n = vec3(0, 0, 0);
      it.p = vec3(0, 0, 0);
      it.t = 10000.0;

      sphere[0].center = vec3(-2.0, 0.0, -3.5);
      sphere[0].radius = 0.5;
      sphere[1].center = vec3(-0.5, 0.0, -3.0);
      sphere[1].radius = 0.5;
      sphere[2].center = vec3(1.0, 0.0, -2.2);
      sphere[2].radius = 0.5;
      plane.p = vec3(0, -0.5, 0);
      plane.n = vec3(0, 1.0, 0);

      Intersect(ray, it);

      if (it.t < 1e3)
      {
        fragColor += computeAO(it);
      }
    }
  }

  uint tileWidth = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  uint pxoff = uint(tileWidth * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x);
  uint v = uint(255.0 * fragColor / (subSamples * subSamples));
  imageData[pxoff] = (0xff << 24) | (v << 16) | (v << 8) | v;
}
