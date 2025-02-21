-- ==
-- tune compiled input @ data/variant-data/2048.in
-- output @ data/variant-data/2048.out
-- notune compiled input @ data/variant-data/1024.in
-- output @ data/variant-data/1024.out
-- tune compiled input @ data/variant-data/512.in
-- output @ data/variant-data/512.out
-- notune compiled input @ data/variant-data/256.in
-- output @ data/variant-data/256.out
-- tune compiled input @ data/variant-data/128.in
-- output @ data/variant-data/128.out
-- notune compiled input @ data/variant-data/64.in
-- output @ data/variant-data/64.out
-- tune compiled input @ data/variant-data/32.in
-- output @ data/variant-data/32.out


let lg (n: i32) =
    let r = 0
    let (_, res) =
      loop (n, r) while n > 1 do
        (n / 2, r + 1i32)
    in res

let matmul [n][m][q] (A: [n][m]f32) (B: [m][q]f32) : [n][q]f32 =
    map (\ Arow ->
            map (\ Bcol ->
                    map2 (*) Arow Bcol
                    |> reduce (+) 0.0f32
                ) (transpose B)
        ) A 

let main [n] (A : [n][n]f32) (B : [n][n]f32) =
    let ks  = map (\i -> 1<<i) (iota (1i32+(lg n)))
    let M   = []
    let (res, _) =
        loop (M,A) for k in ks do -- k = 1, 2, 4, 8, 16, 32, 64, 128
            let A' = unflatten (n/k) (n*k) <| flatten A -- size of A'' is n/k_{i-1} x n/k_{i-1}
            let B' = unflatten (n*k) (n/k) <| flatten B
            let M  = matmul A' B' -- n/k x n/k
            let A''= map2(\Arow Mrow -> 
                            map(\j ->
                                    if j < n/k then Arow[j] + unsafe Mrow[j] else Arow[j]
                               ) (iota (n*k))
                         ) A' M
            in  (M, A'')
    in  res

