# Write UTF-8 no BOM prompt file for llama-cli
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$prompt = "你好，请用中文简短介绍一下你自己（30字以内）。"
[System.IO.File]::WriteAllText("F:\thetrueai\logs\test_prompt.txt", $prompt, $utf8NoBom)
Write-Host "Prompt file written: $((Get-Item F:\thetrueai\logs\test_prompt.txt).Length) bytes"

# Verify content
$bytes = [System.IO.File]::ReadAllBytes("F:\thetrueai\logs\test_prompt.txt")
Write-Host "First 3 bytes (should not be EF BB BF):"
($bytes[0..2] | ForEach-Object { '{0:X2}' -f $_ }) -join ' '
