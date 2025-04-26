import torch

def rotate(seq_t,angle):
    rotate_matrix_row1=torch.cat([torch.cos(angle).unsqueeze(-1),-torch.sin(angle).unsqueeze(-1)],dim=-1)
    rotate_matrix_row2 = torch.cat([torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)], dim=-1)
    rotate_matrix=torch.cat([rotate_matrix_row1.unsqueeze(-1),rotate_matrix_row2.unsqueeze(-1)],dim=-1).to(seq_t.device)
    rotate_seq_t=torch.einsum('ijk,ikl->ijl',seq_t,rotate_matrix)
    return rotate_seq_t

def normalize_x(x_st_t):
    last_dis = x_st_t[:, -2, :]
    last_dis_tan = x_st_t[:, -2, 1] / x_st_t[:, -2, 0]
    last_angle = torch.arctan(last_dis_tan)
    rotate_x_st_t=rotate(seq_t=x_st_t,angle=last_angle)
    return rotate_x_st_t

def normalize_y(y_st_t,last_angle):
    rotate_y_st_t=rotate(seq_t=y_st_t,angle=last_angle)
    return rotate_y_st_t


