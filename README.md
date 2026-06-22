# Chute_Analysis
Analysis of 2026 CMU Buggy chute drone footage

Drone footage can be found at [https://drive.google.com/drive/u/1/folders/15m9A3ij1rXQ4sUsIxHjNfHK-sBj5np0W](https://drive.google.com/drive/u/1/folders/15m9A3ij1rXQ4sUsIxHjNfHK-sBj5np0W) courtesy of Wade Gordon

However, you don't need the drone footage to view the data yourself. All of the data is in *chute_analytics_results.json*
You can just run *line_visalization.html* on a local server like `python -m http.server` if you would like to visually analyze the data

If you want to use the same chute zone boundaries as me, you can find them in *chute_config.json*. If you don't want to use the same zones, delete the *chute_config.json* file and the *line_tracking.py* program will prompt you to make your own. You may have to change the signs of `side_entrance < 0` and `side_exit < 0` depending on the zones you choose.
