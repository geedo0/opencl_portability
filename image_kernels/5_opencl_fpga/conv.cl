#define N       1024
#define CONV_N  3

__kernel void conv(__global ushort * restrict x, __global ushort * restrict y, __global float * restrict h) {
    int iii, jjj;
    float px;
    int o_x, o_y;
    int ii = get_global_id(0);
    int jj = get_global_id(1);
    //Set the new origin
    o_x = jj - 1;
    o_y = ii - 1;
    px = 0;
    for(iii=0; iii<CONV_N; iii++) {
    for(jjj=0; jjj<CONV_N; jjj++) {
      if(((o_y+iii) < 0) || ((o_y+iii) >= N) || 
        ((o_x+jjj) < 0) || ((o_x+jjj) >= N)) {
        //For now, boundary elements take on the value of the origin
        px += ((float) x[(o_y+1)*N + (o_x+1)])*h[iii*CONV_N+jjj];
      }
      else {
        px += ((float) x[(o_y+iii)*N + (o_x+jjj)])*h[iii*CONV_N+jjj];
      }
    }
    }
    px = px < 0 ? 0 : px;
    px = px > 65535 ? 65535 : px;
    y[ii*N+jj] = (ushort) px;

}