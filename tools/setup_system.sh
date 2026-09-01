#!/usr/bin/env bash
#
sysctl -w kernel.numa_balancing=0 || true
