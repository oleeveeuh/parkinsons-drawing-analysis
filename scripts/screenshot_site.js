// Screenshot the pd-drawings gallery card and dialog sections.
const { chromium } = require('/tmp/pwtest/node_modules/playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 950 } });
  await page.goto('http://localhost:8123/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);

  // gallery card
  const card = page.locator('.project-card[data-project="pd-drawings"]');
  await card.screenshot({ path: '/tmp/pd_shot_card.png' });

  // open the dialog
  await card.click();
  await page.waitForTimeout(1500);
  const dialog = page.locator('#project-dialog');
  await dialog.screenshot({ path: '/tmp/pd_shot_dialog_top.png' });

  // scroll to the story start
  await page.evaluate(() => { document.querySelector('#project-dialog').scrollTop = 700; });
  await page.waitForTimeout(800);
  await dialog.screenshot({ path: '/tmp/pd_shot_dialog_story1.png' });

  // scroll to a mid-story figure (split impact)
  await page.evaluate(() => {
    const imgs = [...document.querySelectorAll('#dialog-content .story-figure img')];
    const t = imgs[2];
    document.querySelector('#project-dialog').scrollTop = t.offsetTop - 80;
  });
  await page.waitForTimeout(1200);
  await dialog.screenshot({ path: '/tmp/pd_shot_dialog_story2.png' });

  // scroll to the takeaway
  await page.evaluate(() => {
    const d = document.querySelector('#project-dialog');
    d.scrollTop = d.scrollHeight;
  });
  await page.waitForTimeout(800);
  await dialog.screenshot({ path: '/tmp/pd_shot_dialog_end.png' });

  console.log('screenshots written');
  await browser.close();
})();
