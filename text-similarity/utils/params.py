class FT_Configer():
 
    def __init__(self, params_dict: dict):

        self.hidden_size = params_dict['hidden_size']
        self.pooler_type = params_dict['pooler_type']
        self.unsupervised = params_dict['unsupervised']
        self.evalution_task = params_dict['evalution_task']
        self.learning_rate = params_dict['learning_rate']
        self.epoch = params_dict['epoch']
        self.gradient_acc = params_dict['gradient_acc']
        self.batch_size = params_dict['batch_size']
        self.max_len = params_dict['max_len']
        self.model_save_dir = params_dict['model_save_dir']
        self.warmup_rate = params_dict['warmup_rate']
        self.weight_decay = params_dict['weight_decay']
        self.data_dir = params_dict['data_dir']
        self.model_path = params_dict['model_path']
        self.model_type = params_dict['model_type']
        self.cl_temperature = params_dict['cl_temperature']
