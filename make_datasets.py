from utils import load_model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def load_problem(args):
    """
    a slightly different load_data function that only loads the problem coefficients without training data
    """
    def calc_W_proj(A):
        chunk = torch.mm(A.t(), torch.inverse(torch.mm(A, A.t())))
        Wz_proj = torch.eye(A.shape[-1]).to(device) - torch.mm(chunk, A)
        Wb_proj = chunk
        return Wz_proj, Wb_proj

    if args.dataset in ['DCOPF', 'DCOPF_large']:
        A_primal = torch.load('./data/' + args.dataset + '/A_primal.pt').to(device)
        c_primal = torch.load('./data/' + args.dataset + '/c_primal.pt').to(device)
        WzProj_primal, WbProj_primal = calc_W_proj(A_primal)

        A_dual = torch.load('./data/' + args.dataset + '/A_dual.pt').to(device)
        b_dual = torch.load('./data/' + args.dataset + '/b_dual.pt').to(device)
        WzProj_dual, WbProj_dual = calc_W_proj(A_dual)

        data_dict['A_primal'] = A_primal
        data_dict['c_primal'] = c_primal
        data_dict['WzProj_primal'] = WzProj_primal
        data_dict['WbProj_primal'] = WbProj_primal

        data_dict['A_dual'] = A_dual
        data_dict['b_dual'] = b_dual
        data_dict['WzProj_dual'] = WzProj_dual
        data_dict['WbProj_dual'] = WbProj_dual




def make_dataset(args):
    data = load_problem(args)
    model = load_model(args, data)
    print(f'----- {args.model_id} in {args.framework} framework -----')
    print('#params:', sum(p.numel() for p in model.parameters()))