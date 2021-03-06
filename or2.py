import pandas as pd

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot

import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)


def main(data, eta, epochs, filename, plotFileName):

   

    df = pd.DataFrame(data)

    logging.info(f"This is the actual dataframe(df)")

    df


    X,y = prepare_data(df)

    

    model_OR = Perceptron(eta=eta, epochs=epochs)
    model_OR.fit(X, y)

    _ = model_OR.total_loss()


    save_model(model_OR, filename=filename)
    save_plot(df, plotFileName, model_OR)


if __name__ == '__main__':


    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10


    main(data=OR, eta=ETA, epochs=EPOCHS, filename="or2.model",plotFileName="or2.png")