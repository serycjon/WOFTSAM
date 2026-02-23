from flatsam.config import Config

def get_config():
    conf = Config()
    conf.size = 'tiny'
    conf.memory_stride = 1
    conf.do_not_update_when_not_present = False

    return conf
