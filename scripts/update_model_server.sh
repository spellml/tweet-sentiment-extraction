#!/bin/bash
# This script performs a rolling update of a Spell model server.
set -ex

ls $PWD/tweet-sentiment-extraction/servers/
spell server update --from-file $PWD/tweet-sentiment-extraction/servers/config.yaml