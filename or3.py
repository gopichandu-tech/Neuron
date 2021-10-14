import pandas as pd

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot

import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir ="logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")




def main(data, eta, epochs, filename, plotFileName):

   

    df = pd.DataFrame(data)

    logging.info(f"This is the actual dataframe{df}")

    #df


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
    EPOCHS = 100

    try:
        logging.info(">>>> Starting Traning >>>>")

    
        main(data=OR, eta=ETA, epochs=EPOCHS, filename="or3.model",plotFileName="or3.png")
        logging.info("<<<< Traning Ended Sucessfuly <<<</n")
        
        
    except Exception as e:
        logging.info(Exception(e))
        raise e
