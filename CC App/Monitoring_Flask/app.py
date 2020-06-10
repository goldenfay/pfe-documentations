import re
import time
import traceback
from flask import Flask,render_template

app=Flask(__name__,static_url_path='', 
            static_folder='assets',
            template_folder='templates')

  
@app.route('/view')
def hello():
    return render_template('view.html')

# Running the server
if __name__ == '__main__':
    app.run(debug=True)

   

