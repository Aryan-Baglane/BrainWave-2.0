# Deploy to GitHub - DataClean Pro CLI
# Run: .\deploy.ps1

Write-Host "🚀 DataClean Pro - GitHub Deployment Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Configure Git
Write-Host "✓ Configuring Git..." -ForegroundColor Green
git config user.name "Himanshu Moury" -ErrorAction SilentlyContinue
git config user.email "himanshumoury@example.com" -ErrorAction SilentlyContinue

# Step 2: Show status
Write-Host "✓ Current commits:" -ForegroundColor Green
git log --oneline | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
Write-Host ""

# Step 3: Check remote
try {
    $remote = git remote get-url origin 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Remote configured:" -ForegroundColor Green
        git remote -v
    } else {
        Write-Host "⚠ Remote not configured." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Remote not configured." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📝 GITHUB SETUP INSTRUCTIONS:" -ForegroundColor Magenta
Write-Host "================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "1. Go to: https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: CLI" -ForegroundColor White
Write-Host "3. Leave EMPTY (no README, .gitignore, license)" -ForegroundColor White
Write-Host "4. Click 'Create repository'" -ForegroundColor White
Write-Host ""
Write-Host "5. Then run:" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/HIMANSHUMOURYADTU/CLI.git" -ForegroundColor Yellow
Write-Host "   git branch -M main" -ForegroundColor Yellow
Write-Host "   git push -u origin main" -ForegroundColor Yellow
Write-Host ""
Write-Host "✓ Repository ready to deploy!" -ForegroundColor Green
