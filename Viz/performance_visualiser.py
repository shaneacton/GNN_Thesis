from Code.Main.sys_arg_launcher import get_args
from Code.Utils.model_utils import get_model
from Config.config import get_config

args = get_args()
conf = get_config(args.model_conf, args.train_conf, 0, args)

model, optimizer, scheduler = get_model(conf.model_name)


