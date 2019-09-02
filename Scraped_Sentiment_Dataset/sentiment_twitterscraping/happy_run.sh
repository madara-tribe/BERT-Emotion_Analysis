#!/bin/bash
twitterscraper 😄 --lang ja -o happy.json &
twitterscraper 😆 --lang ja -o happy2.json &
twitterscraper 😂 --lang ja -o happy3.json &

wait;

echo "Done!:twitterscraper"
