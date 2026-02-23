from flatsam.config import Config

def get_config():
    conf = Config()
    conf.size = 'tiny'
    conf.memory_stride = 5
    conf.do_not_update_when_not_present = True

    return conf
