def get_model(model_name, input_size, output_size, hidden_size, batch_size, num_layers, dropout, bias):
    if type(model_name) is not str or model_name not in ["Elman", "LSTM", "GRU", "GBRT", "CNN", "CRNN"]:
        raise ValueError("Model name must be a string and match one of the implemented networks.")
    if type(input_size) is not int or input_size < 0:
        raise ValueError("Input size must be of type int and positive.")
    if type(output_size) is not int or output_size < 0:
        raise ValueError("Output size must be of type int and positive.")
    if type(hidden_size) is not int or hidden_size < 0:
        raise ValueError("Hidden size must be of type int and positive.")
    if type(batch_size) is not int or batch_size < 0:
        raise ValueError("Batch size must be of type int and positive.")
    if type(num_layers) is not int or num_layers < 0:
        raise ValueError("Number of layers must be of type int and positive.")
    if num_layers > 10:
        raise ValueError("Number of layers is unreasonably high.")
    if type(dropout) is not float or dropout < 0 or dropout >= 1:
        raise ValueError("Dropout must be of type float and between 0 and 1.")
    if type(bias) is not bool:
        raise ValueError("Bias must be of type boolean.")

    if model_name == 'Elman':
        from Decoder.src.networks.Elman import Elman
        return Elman()

    elif model_name == 'LSTM':
        from Decoder.src.networks.LSTM import LSTM
        return LSTM()

    elif model_name == 'GRU':
        from Decoder.src.networks.GRU import GRU
        return GRU()

    elif model_name == 'CNN':
        from Decoder.src.networks.CNN import CNN
        return CNN()

    elif model_name == 'CRNN':
        from Decoder.src.networks.CRNN import CRNN
        return CRNN()

    elif model_name == 'Custom':
        from Decoder.src.networks.Custom import Custom
        return Custom()

