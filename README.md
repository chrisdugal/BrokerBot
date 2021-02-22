# BrokerBot

## Overview
BrokerBot is a Slack App that provides users a personal stock trading assistant! BrokerBot can keep track of your watchlist, help you out with technical analysis, and even predict future stock prices.

Disclaimer: Invest at your own discretion, BrokerBot's recommendations should not be taken as professional advice

## Demo

If you would like to try out BrokerBot you can join the [demo workspace](https://join.slack.com/t/brokerbotdemo/shared_invite/zt-mn10vknp-nn1CE4yn0q1b8XP~EpHHGg)

## Installation

First, start by closing the repository:

```
git clone https://github.com/chrisdugal/BrokerBot.git
```

I recommend using a virtual environment for development:

- Start by installing `virtualenv` if you don't have it
```
pip install virtualenv
```

- Once installed, access the project's root directory
```
cd BrokerBot
```

- Create a virtual environment
```
virtualenv brokerBot
```

- Enable the virtual environment
```
source venv/bin/activate
```

- Install the python dependencies on the virtual environment
```
pip install -r requirements.txt
```

## Setup

### Slack App

For instructions on setting up a Slack App, I recommend 1:50 - 5:25 of this [youtube video](https://youtu.be/KJ5bFv-IRFM?t=110)

Under Features > OAuth & Permissions, add the following bot token scopes: 
- app_mentions:read
- chat:write
- files:write
- incoming-webhook

and the following user token scope: 
- files:write

Install the app to your workspace and copy both access tokens at the top of the Features > OAuth & Permissions page as well as your signing secret, which can be found at Settings > Basic Information (you will need these for configuration)

### MongoDB

Create a MongoDB Atlas Cluster

- Firstly, make a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas)

- Choose an Organization Name and Project Name

- Select 'Shared Clusters' (free) and choose any cloud provider - I suggest Google Cloud as they have the best free plan

- Name your cluster and wait a few minutes for the cluster to be created

- Once it's ready, click 'Connect' and follow the on-screen instructions. Once you are asked to choose a connection method, select 'connect your application', then `Python` and `3.6 or later`

- Copy the connection string, you'll need it for configuration (be sure to replace `<password>` and `<dbname>` with your password and "BrokerBot")

### ngrok

Install [ngrok](https://ngrok.com/) to create an HTTP tunnel
- For instructions, I recommend 2:20 - 3:33 of this [youtube video](https://youtu.be/6gHvqXrfjuo?t=140)

Start ngrok
```
./ngrok http 5000
```
or
```
ngrok http 5000
```
Copy the http/https forwarding address, you'll need it later

## Configuration

Finally, make a copy of the `sample-config.json` file named `config.json`
```
cp sample-config.json config.json
```

Fill in the variables with the values from previous steps

## Run BrokerBot

From the project's root directory:

Start BrokerBot
```
python brokerBot.py
```

Connect ngrok address to Slack

- Go to the [slack api website](https://api.slack.com/apps/), navigate to Features > Event Subscriptions and enable events

- Under 'Subscribe to bot events', add the `app_mention` bot user event 

- Paste the address provided by ngrok with "/slack/events" appended and click 'Verify' (you will have to do this every time)

- Save your changes

Finally, add the app to a slack channel and type `@BrokerBot help` to see what Brokerbot can do! 

Enjoy!

