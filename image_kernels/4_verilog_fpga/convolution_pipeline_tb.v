module convolution_pipeline_tb;

wire [15:0] y_out;

reg clk;
reg rst;
reg [15:0] x_in;

integer i;

/*
	Bottom Sobel Filter
	-1,-2,-1,
	0,0,0,
	1,2,1
*/

initial begin
	clk = 0;
	rst = 1;
	x_in = 0;
	#20
	rst = 0;
	//Start stimulus
	//y_out = 16'd40 = 16'h28
	for(i=1; i<20; i=i+1) begin
		#20
		x_in = i;
	end
	//Test saturation of negative value to 0
end

always
	#10 clk = !clk;

convolution_pipeline #(5) UUT(
	.y_out(y_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in),
	.w0(-17'sh1),
	.w1(-17'sh2),
	.w2(-17'sh1),
	.w3(17'sh0),
	.w4(17'sh0),
	.w5(17'sh0),
	.w6(17'sh1),
	.w7(17'sh2),
	.w8(17'sh1),
);

endmodule
