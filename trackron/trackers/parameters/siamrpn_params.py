from trackron.structures import TrackerParams


def get_siamrpn_params():
  params = TrackerParams()
  params.instance_size = 255
  params.exemplar_size = 127
  params.stride = 8
  params.base_size = 8
  params.ratios = [0.33, 0.5, 1, 2, 3]
  params.scales = [8]
  params.anchor_nums = 5
  params.context_amount = 0.5
  params.window_influence = 0.5
  params.penalty_k = 0.24
  params.lr = 0.25
  params.score_size = (params.instance_size - params.exemplar_size
                      ) // params.stride + 1 + params.base_size
  return params