from flatsam.config import Config

def get_config():
    conf = Config()
    conf.model = 'raft'
    conf.checkpoint = 'sintel'

    return conf
