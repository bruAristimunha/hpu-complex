/* Probe v3: shuffle splat semantics.
 * input[0..63] = arange. For lanes p in {0, 1, 31, 32, 33, 63}:
 * splat lane p and store into output chunk p/8 at position p%8 * 64.
 * Simpler: output chunk c (0..5) = splat of lane P[c], P = {0,1,31,32,33,63}.
 * If splat works, each output chunk is constant P[c].
 */
void main(tensor input, tensor lut, tensor output)
{
    int5 coords = { 0, 0, 0, 0, 0 };
    float64 xv = v_f32_ld_tnsr_b(coords, input);
    const int P[6] = { 0, 1, 31, 32, 33, 63 };
    for (int c = 0; c < 6; c++)
    {
        int5 lc = { 0, P[c], 0, 0, 0 };
        uchar256 lutv = v_u8_ld_tnsr_b(lc, lut);
        float64 s = v_f32_shuffle_b(xv, lutv, 0, xv);
        int5 oc = { c * 64, 0, 0, 0, 0 };
        v_f32_st_tnsr(oc, output, s);
    }
}
