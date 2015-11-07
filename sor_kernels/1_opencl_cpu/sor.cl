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