#!/bin/bash
# This script performs a rolling update of a Spell model server.
set -ex

ls $PWD/servers/
spell server update --from-file $PWD/servers/config.yaml