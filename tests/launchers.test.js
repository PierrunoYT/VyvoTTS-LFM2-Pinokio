const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const launcher = require('../pinokio.js')

function menu(files = [], running = [], url) {
  return launcher.menu({}, {
    exists: file => files.includes(file),
    running: file => running.includes(file),
    local: () => url ? { url } : undefined,
  })
}

test('incomplete environment offers Install instead of Start', async () => {
  const items = await menu(['app/env'])
  assert.equal(items[0].href, 'install.js')
  assert.equal(items[0].default, true)
})

test('completed installation offers Start and valid maintenance actions', async () => {
  const items = await menu(['app/env/.installed'])
  assert.equal(items[0].href, 'start.js')
  for (const item of items) assert.ok(fs.existsSync(path.join(__dirname, '..', item.href)))
})

test('maintenance remains visible after completion marker is removed', async () => {
  for (const action of ['install', 'update', 'reset', 'link']) {
    const items = await menu([], [`${action}.js`])
    assert.equal(items[0].href, `${action}.js`)
    assert.equal(items[0].default, true)
  }
})

test('running app transitions from terminal to captured Web UI', async () => {
  const files = ['app/env/.installed']
  assert.equal((await menu(files, ['start.js']))[0].href, 'start.js')
  assert.equal((await menu(files, ['start.js'], 'http://127.0.0.1:7861'))[0].href,
    'http://127.0.0.1:7861')
})

test('startup captures the numeric URL as group one', () => {
  const start = require('../start.js')
  const event = start.run[0].params.on[0]
  const regex = new RegExp(event.event.slice(1, -1))
  assert.equal(regex.exec('Running on local URL:  http://127.0.0.1:7861')[1], 'http://127.0.0.1:7861')
  assert.equal(start.run[1].params.url, '{{input.event[1]}}')
  assert.equal(event.done, true)
  assert.equal(start.daemon, true)
})

test('install verifies dependencies before writing completion marker', () => {
  const steps = require('../install.js').run
  assert.equal(steps[0].method, 'fs.rm')
  assert.equal(steps[0].params.path, 'app/env/.installed')
  const checkIndex = steps.findIndex(step => step.params.message?.includes('uv pip check'))
  assert.ok(checkIndex > steps.findIndex(step => step.params.uri === 'torch.js'))
  assert.equal(steps.at(-1).method, 'fs.write')
  assert.equal(steps.at(-1).params.path, 'app/env/.installed')
  assert.ok(checkIndex < steps.length - 1)
  assert.equal(require('../reset.js').run[0].params.path, 'app/env')
})

test('update refreshes dependencies after fast-forward pull', () => {
  const steps = require('../update.js').run
  assert.equal(steps[0].params.message, 'git pull --ff-only')
  assert.equal(steps[1].params.uri, 'install.js')
})

test('each supported platform selects one torch backend; Windows AMD uses CPU', () => {
  const steps = require('../torch.js').run
  for (const platform of ['win32', 'darwin', 'linux']) {
    for (const gpu of ['nvidia', 'amd', 'apple', undefined]) {
      const matches = steps.filter(step =>
        Function('platform', 'gpu', `return (${step.when.slice(2, -2)})`)(platform, gpu))
      assert.equal(matches.length, 1, `${platform}/${gpu}`)
      if (platform === 'win32' && gpu !== 'nvidia') {
        const messages = matches[0].params.message
        assert.equal(messages[0], 'uv pip uninstall torch-directml')
        assert.match(messages[1], /whl\/cpu/)
        assert.doesNotMatch(messages[1], /directml/)
      }
    }
  }
})
