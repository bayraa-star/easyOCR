from roboflow import Roboflow
rf = Roboflow(api_key="zDuF0k8jn0OqLKvQSCY3")
project = rf.workspace("coo-sntjh").project("vehicle-brand-color-side")
version = project.version(2)
dataset = version.download("multiclass")
