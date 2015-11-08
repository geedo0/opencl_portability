module n_shift_register_tb;

wire [16:0] out;
reg clk;
reg rst;
reg [16:0] in;

initial begin
	clk <= 0;
	rst <= 1;
	in <= 0;
	#20
	rst <= 0;
	//Stimulus
	in <= 10;
	#20
	in <= 20;
	#20
	in <= 0;
end

always
	#10 clk<=!clk;

n_shift_register #(4)UUT(
	.out(out),
	.clk(clk),
	.rst(rst),
	.in(in)
);

endmodule
