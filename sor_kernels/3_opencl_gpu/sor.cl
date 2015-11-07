__kernel void sor(const int N, __global float* A, __global float* B) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int rowN = N;

    //If the row and col values are not boundary elements, compute the average
    if(row > 0 && row < (N-1) && col > 0 && col < (N-1)) {
        B[row*rowN+col] = (
            A[row*rowN + col-1] +
            A[row*rowN + col] +
            A[row*rowN + col+1] +
            A[(row-1)*rowN + col] +
            A[(row+1)*rowN + col]) * .2;
    }
    //else the element does not change
    else {
        B[row*rowN + col] = A[row*rowN + col];
    }
}

__kernel void sor_no_branch(const int N, __global float* A, __global float* B) {
    int row = get_global_id(0) + 1;
    int col = get_global_id(1) + 1;
    int rowN = N;
    B[row*rowN+col] = (
        A[row*rowN + col-1] +
        A[row*rowN + col] +
        A[row*rowN + col+1] +
        A[(row-1)*rowN + col] +
        A[(row+1)*rowN + col]) * .2;
}

__kernel void sor_shared_mem(const int N, __global float* A, __global float* B)
{
    int row = get_global_id(0) + 1;
    int col = get_global_id(1) + 1;
    int rowN = N;

    float acc;
    __local float aTile[16][16];
    aTile[get_local_id(1)][get_local_id(0)] = A[row*rowN + col];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0)>0 && get_local_id(0)<(get_local_size(0)-1) && get_local_id(1)>0 && get_local_id(1)<(get_local_size(1)-1)) {
        acc = (
            aTile[get_local_id(1)][get_local_id(0)+1]+
            aTile[get_local_id(1)][get_local_id(0)]+
            aTile[get_local_id(1)][get_local_id(0)-1]+
            aTile[get_local_id(1)+1][get_local_id(0)]+
            aTile[get_local_id(1)-1][get_local_id(0)]) * 0.2;
    }
    else {
        float acc = aTile[get_local_id(1)][get_local_id(0)];
        if(get_local_id(0)==0)
            acc += A[row*rowN + col-1];
        else
            acc += aTile[get_local_id(1)][get_local_id(0)-1];
        if(get_local_id(0)==get_local_size(0))
            acc += A[row*rowN + col+1];
        else
            acc += aTile[get_local_id(1)][get_local_id(0)+1];
        if(get_local_id(1)==0)
            acc += A[(row-1)*rowN + col];
        else
            acc += aTile[get_local_id(1)-1][get_local_id(0)];
        if(get_local_id(1)==get_local_size(1))
            acc += A[(row+1)*rowN + col];
        else
            acc += aTile[get_local_id(1)+1][get_local_id(0)];
        acc*0.2;
    }
    B[row*rowN+col] = (float) acc;
    
}
