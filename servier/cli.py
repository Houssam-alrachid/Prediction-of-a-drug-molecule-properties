import sys
import click

from servier.src.main import Train, Predict, Evaluate
import servier.src.config as config

@click.group()
@click.version_option("1.0.0")
def main():
    """Drug molecule properties prediction app"""
    pass


@main.command()
@click.option('--model_type', '-d', type=str, required=True, default='FF', help="Please enter the name of the model you want to train")
def train(model_type):
    """Train a Deep learning model for prediction and save the pretrained model to disk"""
    click.echo(Train(model_type))


@main.command()
@click.option('--path_x_test', '-p', type=str, required=True, default='servier/data/dataset_single.csv', help="Please enter the path of data in order to perform prediction")
@click.option('--model_type', '-p', type=str, required=True, default='FF', help="Please enter the model name")

def predict(path_x_test, model_type):
    """Perform prediction using a pretrained Deep Learning prediction model"""
    click.echo(Predict(path_x_test, model_type))


@main.command()
@click.option('--y_test', '-t', type=float, required=True, help='Enter the true array')
@click.option('--y_pred', '-p', type=float, required=True, help='Enter the predicted array')
def evaluate(y_test, y_pred):
    """Evaluate the prediction model"""
    click.echo(Evaluate(y_test, y_pred))
