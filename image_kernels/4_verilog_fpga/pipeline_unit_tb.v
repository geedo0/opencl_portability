module pipeline_unit_tb;

wire [16:0] y_out;

reg clk;
reg rst;
reg [16:0] x_in;
reg [16:0] y_in;
reg [16:0] w_in;

initial begin
	clk <= 0;
	rst <= 1;
	x_in <= 0;
	y_in <= 0;
	w_in <= 0;
	#20
	rst <= 0;
	//Begin test cases
	//y_out = 0+0*0 = 0
	#20
	y_in <= 0;
	x_in <= 0;
	w_in <= 0;
	//y_out = 1+2*4 = 9
	#20
	y_in <= 1;
	x_in <= 2;
	w_in <= 4;
	//y_out = -1+(-1)(-1) = 0
	#20
	y_in <= -1;
	x_in <= -1;
	w_in <= -1;
	//y_out = (-1)+2*4 = 7
	#20
	y_in <= -1;
	x_in <= 2;
	w_in <= 4;
	//y_out = 1+(-2)*4 = -7
	#20
	y_in <= 1;
	x_in <= -2;
	w_in <= 4;
	//y_out = 1+2*(-4) = -3
	#20
	y_in <= 1;
	x_in <= 2;
	w_in <= -2;
	//y_out = 0+2*(0x0ffff) = 0x0ffff (saturated)
	#20
	y_in <= 0;
	x_in <= 17'h0ffff;
	w_in <= 2;
	//y_out = 0x0ffff+0x0ffff*0x0ffff = 0x0ffff (super saturated)
	#20
	y_in <= 17'h0ffff;
	x_in <= 17'h0ffff;
	w_in <= 17'h0ffff;
	//y_out = -2^(16)-2^16(2^16-1) = -2^16 (Negatively saturated)
	#20
	y_in <= 17'h10000;
	x_in <= 17'h10000;
	w_in <= 17'h0ffff;
end

always
	#10 clk<=!clk;

pipeline_unit UUT(
	.y_out(y_out),
	.clk(clk),
	.rst(rst),
	.x_in(x_in),
	.y_in(y_in),
	.w_in(w_in)
);

endmodule