# rat-inference

T1000 for rats.


## Pi Setup
`curl -LsSf https://astral.sh/uv/install.sh | sh`

```
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-libcamera libcamera-apps
sudo usermod -aG video $USER   # then log out/in once or reboot
```


```
cd ~/rat-inference
deactivate 2>/dev/null || true
rm -rf .venv
uv venv -p /usr/bin/python3.11 --system-site-packages
source .venv/bin/activate
python -V    # should show 3.11.x
```


```
uv pip install -U pip wheel setuptools
uv pip install picamera2 ultralytics
```

```
python -c "import libcamera; print('libcamera OK:', libcamera.__file__)"
python -c "from picamera2 import Picamera2; print('picamera2 OK')"
python -c "import ultralytics, torch; print('ultralytics OK, torch', torch.__version__)"
```

## Running
`python rt_100.py`
