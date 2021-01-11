#!/bin/bash
# This script performs a rolling update of a Spell model server.
set -ex

pushd ../ && REPO_ROOT=$PWD && popd
spell server update --from-file $REPO_ROOT/servers/config.yaml