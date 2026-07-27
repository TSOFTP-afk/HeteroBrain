# Extract Jinja chat_template from MiniCPM5-1B GGUF metadata.
# GGUF format: key = uint64(len) + bytes; value = uint32(type) + typed_value
# For string type (8): uint64(len) + bytes
# All script content in ASCII; output file is UTF-8.
$ErrorActionPreference = "Stop"

$gguf = "F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf"
$outFile = "F:\hb_models\minicpm5-chat.jinja"

$bytes = [System.IO.File]::ReadAllBytes($gguf)
Write-Host ("GGUF size: " + $bytes.Length + " bytes") -ForegroundColor Cyan

# Find "tokenizer.chat_template" key (23 bytes)
$key = [System.Text.Encoding]::ASCII.GetBytes("tokenizer.chat_template")
$keyLen = $key.Length  # 23

# Search for the key bytes
$keyIdx = -1
for ($i = 0; $i -lt $bytes.Length - $keyLen; $i++) {
    $match = $true
    for ($j = 0; $j -lt $keyLen; $j++) {
        if ($bytes[$i + $j] -ne $key[$j]) { $match = $false; break }
    }
    if ($match) {
        # Verify preceding 8 bytes are uint64 LE == 23
        $lenBytes = $bytes[($i - 8)..($i - 1)]
        $expectedLen = [BitConverter]::ToUInt64($lenBytes, 0)
        if ($expectedLen -eq $keyLen) {
            $keyIdx = $i
            Write-Host ("[FOUND] key at offset " + $i + ", preceded by len=" + $expectedLen) -ForegroundColor Green
            break
        }
    }
}

if ($keyIdx -lt 0) {
    throw "tokenizer.chat_template key not found in GGUF"
}

# After key bytes: uint32 value_type (4 bytes LE)
$valueTypeOffset = $keyIdx + $keyLen
$valueType = [BitConverter]::ToUInt32($bytes, $valueTypeOffset)
Write-Host ("value_type at offset " + $valueTypeOffset + " = " + $valueType + " (8=string expected)") -ForegroundColor Cyan

if ($valueType -ne 8) {
    throw "expected string type (8), got $valueType"
}

# After value_type: uint64 length (8 bytes LE)
$valueLenOffset = $valueTypeOffset + 4
$valueLen = [BitConverter]::ToUInt64($bytes, $valueLenOffset)
Write-Host ("value_len at offset " + $valueLenOffset + " = " + $valueLen + " bytes") -ForegroundColor Cyan

# After length: value bytes
$valueOffset = $valueLenOffset + 8
$valueEnd = $valueOffset + [int]$valueLen
Write-Host ("value bytes: offset " + $valueOffset + " to " + $valueEnd) -ForegroundColor Cyan

if ($valueEnd -gt $bytes.Length) {
    throw "value extends beyond file end"
}

# Extract and decode as UTF-8
$valueBytes = New-Object byte[] $valueLen
[Array]::Copy($bytes, $valueOffset, $valueBytes, 0, [int]$valueLen)
$jinja = [System.Text.Encoding]::UTF8.GetString($valueBytes)

# Write to .jinja file (UTF-8 no BOM)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($outFile, $jinja, $utf8NoBom)

Write-Host ""
Write-Host ("[OK] wrote " + $valueLen + " bytes to " + $outFile) -ForegroundColor Green
Write-Host ""
Write-Host "===== First 500 chars of chat template =====" -ForegroundColor Cyan
Write-Host $jinja.Substring(0, [Math]::Min(500, $jinja.Length))
Write-Host ""
Write-Host "===== Last 200 chars of chat template =====" -ForegroundColor Cyan
Write-Host $jinja.Substring([Math]::Max(0, $jinja.Length - 200))
