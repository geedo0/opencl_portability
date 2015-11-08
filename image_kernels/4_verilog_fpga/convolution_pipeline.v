module convolution_pipeline(
	y_out,
	clk,
	rst,
	x_in,
	w0,
	w1,
	w2,
	w3,
	w4,
	w5,
	w6,
	w7,
	w8
);

//Defines the expected length of the input image. Min(4)
parameter length = 4;

output [15:0] y_out;

input clk;
input rst;

input [15:0] x_in;
input [16:0] w0;
input [16:0] w1;
input [16:0] w2;
input [16:0] w3;
input [16:0] w4;
input [16:0] w5;
input [16:0] w6;
input [16:0] w7;
input [16:0] w8;

wire [16:0] x_in_ext;
assign x_in_ext = {1'b0, x_in};

wire [16:0] y0_out;
wire [16:0] y1_out;
wire [16:0] y2_out;
wire [16:0] y3_out;
wire [16:0] y4_out;
wire [16:0] y5_out;
wire [16:0] y6_out;
wire [16:0] y7_out;
wire [16:0] y8_out;
assign y_out = $signed(y8_out) < $signed(17'h0) ? 16'h0 : y8_out[15:0];

wire [16:0] r0_out;
wire [16:0] r1_out;

//0,0
pipeline_unit p0(
	.y_out(y0_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(17'h0),
	.w_in(w0)
);

//0,1
pipeline_unit p1(
	.y_out(y1_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(y0_out),
	.w_in(w1)
);

//0,2
pipeline_unit p2(
	.y_out(y2_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(y1_out),
	.w_in(w2)
);

n_shift_register #(length-3) r0(
	.out(r0_out),
	.clk(clk),
	.rst(rst),
	.in(y2_out)
);

//1,0
pipeline_unit p3(
	.y_out(y3_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(r0_out),
	.w_in(w3)
);

//1,1
pipeline_unit p4(
	.y_out(y4_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(y3_out),
	.w_in(w4)
);

//1,2
pipeline_unit p5(
	.y_out(y5_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(y4_out),
	.w_in(w5)
);

n_shift_register #(length-3) r1(
	.out(r1_out),
	.clk(clk),
	.rst(rst),
	.in(y5_out)
);

//2,0
pipeline_unit p6(
	.y_out(y6_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(r1_out),
	.w_in(w6)
);

//2,1
pipeline_unit p7(
	.y_out(y7_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(y6_out),
	.w_in(w7)
);

//2,2
pipeline_unit p8(
	.y_out(y8_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in_ext),
	.y_in(y7_out),
	.w_in(w8)
);

endmodule
