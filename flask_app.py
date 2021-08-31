from flask import Flask, request
import network2
import numpy as np
import ast

app = Flask(__name__)
net = network2.load("./ShanesNetwork0")

@app.route("/digits")
def digits():
    digits = request.args.get('digits')
    newDigits = ast.literal_eval(digits)
    intResults = net.feedforward(newDigits)
    result = np.argmax(intResults)
    return str(result)

if __name__ == "__main__":
    app.run()
