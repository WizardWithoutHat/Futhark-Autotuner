-- | Program for generating various "random" datasets
--   in which the values for "(M, N, n, nanfreq)" are
--   given as input, where "M" denotes the number of pixels,
--   "N" denotes the timeseries length, "n" denotes the
--   length of the training set, and "nanfreq" denotes the
--   frequency of NAN values in the image. 

import "lib/github.com/diku-dk/cpprandom/random"

module distf32 = uniform_real_distribution f32 minstd_rand
module disti32 = uniform_int_distribution  i32 minstd_rand

let genRands (q: i32) : [q]f32 =
  let arr = replicate q 0.0
  let rng = minstd_rand.rng_from_seed [123] in
  let (arr, _) = 
    loop (arr,rng) for i < q do
        let (rng, x) = distf32.rand (1,6) rng
        let arr[i] = x
        in  (arr, rng)
  in arr

-- for example, something similar to sahara dataset can be generated
-- with the arguments:
-- 67968i32 414i32 1i32 3i32 228i32 12.0f32 0.25f32 1.736126f32 0.5f32
let main (M: i32) (N: i32) (n: i32) (nanfreq: f32) :
         (i32, i32, i32, f32, f32, f32, [N]i32, [M][N]f32) =
  let trend = 1i32
  let k     = 3i32
  let freq  = 12f32 -- for peru, 365f32 for sahara
  let hfrac = 0.25f32
  let lam   = 1.736126f32

  -- for simplicity take the mapping indices from 1..N
  let mappingindices = map (+1) (iota N)
  let rngi = minstd_rand.rng_from_seed [246]

  -- initialize the image
  let image = replicate M (replicate N f32.nan)
  let (image, _) =
    loop (image, rngi) for i < M do
        -- init the floating-point seed.
        let rngf     = minstd_rand.rng_from_seed [123+i]
        let rngf_nan = minstd_rand.rng_from_seed [369+i]
        -- compute the break point.
        let (rngi, b0) = disti32.rand (1, N-n-1) rngi
        let break = b0 + n
        -- fill in the time-series up to the breaking point with
        -- values in interval (4000, 8000) describing raining forests.
        let (image, rngf, rngf_nan) =
            loop (image, rngf, rngf_nan) for j < break do
                let (rngf_nan, q) = distf32.rand (0, 1) rngf_nan in
                if q < nanfreq then (image, rngf, rngf_nan)
                else let (rngf, x) = distf32.rand (4000, 8000) rngf
                     let image[i,j] = x
                     in  (image, rngf, rngf_nan)
        -- fill in the points after the break.
        let (image, _rngf, _rngf_nan) =
            loop (image, rngf, rngf_nan) for j0 < N-break do
                let (rngf_nan, q) = distf32.rand (0, 1) rngf_nan in
                if q < nanfreq then (image, rngf, rngf_nan)
                else let j = j0 + break
                     let (rngf, x) = distf32.rand (0, 5000) rngf
                     let image[i,j] = x
                     in  (image, rngf, rngf_nan)
        in  (image, rngi)
  in (trend, k, n, freq, hfrac, lam, mappingindices, image)

