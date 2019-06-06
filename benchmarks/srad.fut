-- Rodinia's SRAD, but extended with another outer level of
-- parallelism, such that multiple SRAD instances are computed in
-- parallel.
--
-- ==
-- tune compiled input @ data/srad-data/train-D1.in
-- output @ data/srad-data/train-D1.out
--
-- tune compiled input @ data/srad-data/train-D2.in
-- output @ data/srad-data/train-D2.out
--
-- tune compiled input @ data/srad-data/train-D3.in
--
-- notune compiled input @ data/srad-data/D1.in
-- output @ data/srad-data/D1.out
--
-- notune compiled input @ data/srad-data/D2.in
-- output @ data/srad-data/D2.out

module srad = import "srad-baseline"

let main [num_images][rows][cols] (images: [num_images][rows][cols]u8) : [num_images][rows][cols]f32 =
  let niter = 100
  let lambda = 0.5
--  in (num_images, rows, cols)
  in map (\image -> srad.do_srad(niter, lambda, image)) images
