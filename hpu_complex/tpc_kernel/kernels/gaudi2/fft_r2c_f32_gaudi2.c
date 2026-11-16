/* Fused FFT (DFT form) for Gaudi2 TPC, one launch, v1.
 *
 * One instance per row. For each output bin k, accumulate vector
 * products over input chunks and reduce horizontally. Only verified
 * intrinsics: ld_tnsr / mul / add / reduce_add / st_tnsr_partial.
 *
 * tensors (TPC geometry, dim0 innermost):
 *   input  [n, rows]     f32   (dim0 = j, dim1 = row)
 *   twid   [n, n, 2]     f32   (dim0 = j, dim1 = k, dim2 = plane cos/sin)
 *   output [n, 2, rows]  f32   (dim0 = k, dim1 = plane, dim2 = row)
 * index space: dim1 = rows.
 */
void main(tensor input, tensor twid, tensor output)
{
    const int rows = get_dim_size(input, 1);
    const int n = get_dim_size(input, 0);
    const int chunks = n / 64;

    // Canonical index-space loop (TPC docs): this invocation owns the
    // contiguous subset [offset, offset+size) of the rows dim.
    const int row_start = get_index_space_offset()[0];
    const int row_end = row_start + get_index_space_size()[0];
    for (int row = row_start; row < row_end && row < rows; row++)
    {
    for (int k = 0; k < n; k++)
    {
        float64 acc_re = {};
        float64 acc_im = {};
        for (int jc = 0; jc < chunks; jc++)
        {
            int5 xc = { jc * 64, row, 0, 0, 0 };
            float64 xv = v_f32_ld_tnsr_b(xc, input);
            int5 wre_c = { jc * 64, k, 0, 0, 0 };
            int5 wim_c = { jc * 64, k, 1, 0, 0 };
            float64 wre = v_f32_ld_tnsr_b(wre_c, twid);
            float64 wim = v_f32_ld_tnsr_b(wim_c, twid);
            acc_re = v_f32_add_b(acc_re, v_f32_mul_b(xv, wre, 0), 0);
            acc_im = v_f32_add_b(acc_im, v_f32_mul_b(xv, wim, 0), 0);
        }
        acc_re = v_f32_reduce_add(acc_re);
        acc_im = v_f32_reduce_add(acc_im);
        int5 ore = { k, 0, row, 0, 0 };
        int5 oim = { k, 1, row, 0, 0 };
        v_f32_st_tnsr_partial(ore, output, acc_re, 1, 0);
        v_f32_st_tnsr_partial(oim, output, acc_im, 1, 0);
    }
    }
}
