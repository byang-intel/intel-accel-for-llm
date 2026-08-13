#!/bin/bash
#
# disable-acs.sh
#
# 遍历系统中所有 PCIe 设备，查找 ACS (Access Control Services)
# Extended Capability，打印当前状态；默认 dry-run，仅在指定
# --apply 时才真正清零 ACSCtl 寄存器（关闭 P2P 请求/完成重定向等）。
#
# 用法:
#   ./disable-acs.sh                 # dry-run，只打印现状和将要做的改动
#   ./disable-acs.sh --apply         # 真正执行修改，并打印修改前后对比
#   ./disable-acs.sh --apply --bdf 0000:17:00.0   # 只对指定设备操作
#
# 需要 root 权限（读写 PCI 配置空间）。
#
set -euo pipefail

ACS_EXT_CAP_ID="000d"   # PCIe Extended Capability ID for ACS
DRY_RUN=1
TARGET_BDF=""

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            DRY_RUN=0
            shift
            ;;
        --bdf)
            TARGET_BDF="$2"
            shift 2
            ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            exit 1
            ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "错误: 需要 root 权限运行 (读写 PCI 配置空间)" >&2
    exit 1
fi

if ! command -v setpci >/dev/null 2>&1; then
    echo "错误: 未找到 setpci，请安装 pciutils" >&2
    exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "==================================================================="
    echo " 当前为 DRY-RUN 模式，只会打印现状与计划变更，不会写入任何寄存器。"
    echo " 确认无误后加 --apply 参数真正执行。"
    echo "==================================================================="
else
    echo "==================================================================="
    echo " 当前为 APPLY 模式，将真正修改 ACSCtl 寄存器！"
    echo "==================================================================="
fi
echo

# 解析 ACSCtl 16 位值，打印各功能位状态
decode_acsctl() {
    local val=$1
    local v=$((16#$val))
    local names=("SrcValid" "TransBlk" "P2pReqRedir" "P2pCmpltRedir" "UpstreamFwd" "EgressCtrl" "DirectTrans")
    local out=""
    for i in "${!names[@]}"; do
        if (( (v >> i) & 1 )); then
            out+="${names[$i]}+ "
        else
            out+="${names[$i]}- "
        fi
    done
    echo "$out"
}

# 在指定设备的 Extended Capability 链表中查找 ACS (0x000d)，
# 返回该 capability 在配置空间中的偏移（十六进制，不带0x前缀），找不到返回空
find_acs_offset() {
    local bdf=$1
    local ptr="100"   # Extended capability 链表从 0x100 开始
    local guard=0

    while [[ "$ptr" != "000" && -n "$ptr" && $guard -lt 64 ]]; do
        guard=$((guard + 1))
        local header
        header=$(setpci -s "$bdf" "${ptr}.l" 2>/dev/null) || break
        [[ -z "$header" || "$header" == "ffffffff" ]] && break

        # header 是小端 32bit hex: 低16位=cap id, 高16位低4位是version，其余是next ptr
        local cap_id="${header: -4}"          # 低16位 = cap id (4 hex chars, 但要小端处理)
        # setpci 输出的是寄存器值的16进制字符串，按大端字符串表示，
        # 实际上 header 格式为 "NNNNIIII" 其中 IIII 是 cap id (小端已由setpci处理)
        cap_id="${header:4:4}"
        local next_and_ver="${header:0:4}"
        local next="${next_and_ver:0:3}"

        if [[ "$cap_id" == "$ACS_EXT_CAP_ID" ]]; then
            echo "$ptr"
            return 0
        fi

        [[ "$next" == "000" ]] && break
        ptr="$next"
    done
    return 1
}

DEVICES=$(lspci -D -d '*:*:*' | awk '{print $1}')
if [[ -n "$TARGET_BDF" ]]; then
    DEVICES="$TARGET_BDF"
fi

FOUND_COUNT=0
CHANGED_COUNT=0

printf "%-14s %-8s %-40s %-40s\n" "BDF" "OFFSET" "BEFORE" "AFTER(计划/实际)"
printf "%-14s %-8s %-40s %-40s\n" "------" "------" "------" "------"

for bdf in $DEVICES; do
    offset=$(find_acs_offset "$bdf" || true)
    [[ -z "$offset" ]] && continue

    FOUND_COUNT=$((FOUND_COUNT + 1))

    # ACSCtl 寄存器位于 capability 起始偏移 + 0x06，宽度2字节
    ctl_offset_dec=$(( 16#$offset + 6 ))
    ctl_offset_hex=$(printf '%x' "$ctl_offset_dec")

    before=$(setpci -s "$bdf" "${ctl_offset_hex}.w" 2>/dev/null) || continue
    before_decoded=$(decode_acsctl "$before")

    devname=$(lspci -s "$bdf" | cut -d' ' -f2-)

    if [[ "$before" == "0000" ]]; then
        printf "%-14s 0x%-6s %-40s %-40s\n" "$bdf" "$ctl_offset_hex" \
            "$before ($before_decoded)" "已是全关，跳过"
        continue
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf "%-14s 0x%-6s %-40s %-40s\n" "$bdf" "$ctl_offset_hex" \
            "$before ($before_decoded)" "将写入 0000 (dry-run，未执行)"
        CHANGED_COUNT=$((CHANGED_COUNT + 1))
    else
        setpci -s "$bdf" "${ctl_offset_hex}.w=0000"
        after=$(setpci -s "$bdf" "${ctl_offset_hex}.w" 2>/dev/null)
        after_decoded=$(decode_acsctl "$after")
        printf "%-14s 0x%-6s %-40s %-40s\n" "$bdf" "$ctl_offset_hex" \
            "$before ($before_decoded)" "$after ($after_decoded)"
        CHANGED_COUNT=$((CHANGED_COUNT + 1))
    fi
    echo "               设备: $devname"
done

echo
echo "==================================================================="
echo "扫描完成: 共发现 $FOUND_COUNT 个带 ACS 能力的设备，涉及变更 $CHANGED_COUNT 个。"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "这是 dry-run 结果，未做任何实际修改。确认无误后运行:"
    echo "    sudo $0 --apply"
else
    echo "已完成写入。注意: 此修改在重启/设备reset后会丢失，"
    echo "如需持久化请配合 systemd service 在开机时自动执行本脚本 --apply。"
fi
echo "==================================================================="
