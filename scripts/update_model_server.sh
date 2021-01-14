#!/bin/bash
# This script performs a rolling update of a Spell model server.
set -ex

spell server update --from-file $PWD/servers/config.yaml