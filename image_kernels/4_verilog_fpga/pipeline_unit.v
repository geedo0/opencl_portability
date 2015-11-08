module pipeline_unit(
	y_out,
	clk,
	rst,
	x_in,
	y_in,
	w_in
);

output reg [16:0] y_out;

input clk;
input rst;

input [16:0] x_in;
input [16:0] y_in;
input [16:0] w_in;

//Sign extend
wire [33:0] x_in_ext;
assign x_in_ext = x_in[16] ? {17'h1ffff, x_in} : x_in;
wire [33:0] y_in_ext;
assign y_in_ext = y_in[16] ? {17'h1ffff, y_in} : y_in;
wire [33:0] w_in_ext;
assign w_in_ext = w_in[16] ? {17'h1ffff, w_in} : w_in;

reg [33:0] y_out_ext;

always@(posedge clk or posedge rst) begin
	if(rst) begin
		y_out <= 0;
	end
	else begin
		//y_out = y_in + x_in*w_in
		y_out_ext = y_in_ext + x_in_ext*w_in_ext;
		y_out = $signed(y_out_ext) > $signed(34'h00000ffff) ? 17'h0ffff :
				$signed(y_out_ext) < $signed(34'h3ffff0000) ? 17'h10000 : y_out_ext[16:0];
	end
end

endmodule
