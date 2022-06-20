from .dimp_params import get_dimp50_params, get_super_dimp_params, get_s3t_params, get_stark_params
from .siamrpn_params import get_siamrpn_params



def get_params(param_name):
  func = globals().get(f'get_{param_name}_params', None)
  if func is None:
    raise NotImplemented(f'params {param_name} not supported')
  return func()
# def get_tracker_params(param_name):
#   if param_name == 'dimp50':
#     return get_dimp50_params()
#   elif param_name == 'super_dimp':
#     return get_super_dimp_params()
#   elif param_name == 's3t':
#     return get_s3t_params()
#   elif param_name == 'stark':
#     return get_stark_params()
#   raise ValueError('tracker name not found')
#   return None