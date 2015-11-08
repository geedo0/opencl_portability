module n_shift_register
#(parameter n=1)
(
	out,
	clk,
	rst,
	in
);

output [16:0] out;

input clk;
input rst;

input [16:0] in;

reg [16:0] registers [0:n-1];

assign out = registers[n-1];

integer i;

always@(posedge clk or posedge rst) begin
	if(rst) begin
		for(i=0; i<n; i=i+1) begin
			registers[i] <= 0;
		end
	end
	else begin
		for(i=1; i<n; i=i+1) begin
			registers[i] <= registers[i-1];
		end
		registers[0] <= in;
	end
end

endmodule
