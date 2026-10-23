/* Personalize: seasonal page backdrop + optional Bible verse bar. Per-browser preferences in localStorage
 * (t58-bg, t58-verse); everything degrades to the default mountains / no verse bar if storage is unavailable.
 * Verses are the King James Version (public domain). The verse is chosen from the clock (30-minute slots), so it is
 * the same on every page and tab without any server state. */
(function () {
  'use strict';
  var BGS = ['mountains', 'spring', 'summer', 'autumn', 'winter'];
  var SLOT_MS = 30 * 60 * 1000;
  var VERSES = [["Psalm 23:1", "The LORD is my shepherd; I shall not want."], ["Proverbs 3:5", "Trust in the LORD with all thine heart; and lean not unto thine own understanding."], ["Philippians 4:13", "I can do all things through Christ which strengtheneth me."], ["Joshua 1:9", "Have not I commanded thee? Be strong and of a good courage; be not afraid, neither be thou dismayed: for the LORD thy God is with thee whithersoever thou goest."], ["Isaiah 41:10", "Fear thou not; for I am with thee: be not dismayed; for I am thy God: I will strengthen thee; yea, I will help thee; yea, I will uphold thee with the right hand of my righteousness."], ["Jeremiah 29:11", "For I know the thoughts that I think toward you, saith the LORD, thoughts of peace, and not of evil, to give you an expected end."], ["Romans 8:28", "And we know that all things work together for good to them that love God, to them who are the called according to his purpose."], ["Psalm 46:1", "God is our refuge and strength, a very present help in trouble."], ["Proverbs 16:3", "Commit thy works unto the LORD, and thy thoughts shall be established."], ["Matthew 6:33", "But seek ye first the kingdom of God, and his righteousness; and all these things shall be added unto you."], ["Psalm 37:5", "Commit thy way unto the LORD; trust also in him; and he shall bring it to pass."], ["Proverbs 21:5", "The thoughts of the diligent tend only to plenteousness; but of every one that is hasty only to want."], ["Colossians 3:23", "And whatsoever ye do, do it heartily, as to the Lord, and not unto men."], ["Proverbs 22:3", "A prudent man foreseeth the evil, and hideth himself: but the simple pass on, and are punished."], ["Psalm 118:24", "This is the day which the LORD hath made; we will rejoice and be glad in it."], ["Isaiah 40:31", "But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles; they shall run, and not be weary; and they shall walk, and not faint."], ["Galatians 6:9", "And let us not be weary in well doing: for in due season we shall reap, if we faint not."], ["2 Timothy 1:7", "For God hath not given us the spirit of fear; but of power, and of love, and of a sound mind."], ["Psalm 121:1-2", "I will lift up mine eyes unto the hills, from whence cometh my help. My help cometh from the LORD, which made heaven and earth."], ["Proverbs 27:17", "Iron sharpeneth iron; so a man sharpeneth the countenance of his friend."], ["Matthew 11:28", "Come unto me, all ye that labour and are heavy laden, and I will give you rest."], ["John 14:27", "Peace I leave with you, my peace I give unto you: not as the world giveth, give I unto you. Let not your heart be troubled, neither let it be afraid."], ["Psalm 27:1", "The LORD is my light and my salvation; whom shall I fear? the LORD is the strength of my life; of whom shall I be afraid?"], ["Proverbs 10:4", "He becometh poor that dealeth with a slack hand: but the hand of the diligent maketh rich."], ["Lamentations 3:22-23", "It is of the LORD's mercies that we are not consumed, because his compassions fail not. They are new every morning: great is thy faithfulness."], ["Psalm 19:14", "Let the words of my mouth, and the meditation of my heart, be acceptable in thy sight, O LORD, my strength, and my redeemer."], ["Romans 12:12", "Rejoicing in hope; patient in tribulation; continuing instant in prayer."], ["James 1:5", "If any of you lack wisdom, let him ask of God, that giveth to all men liberally, and upbraideth not; and it shall be given him."], ["Proverbs 4:7", "Wisdom is the principal thing; therefore get wisdom: and with all thy getting get understanding."], ["Psalm 34:8", "O taste and see that the LORD is good: blessed is the man that trusteth in him."], ["Hebrews 11:1", "Now faith is the substance of things hoped for, the evidence of things not seen."], ["1 Corinthians 16:13", "Watch ye, stand fast in the faith, quit you like men, be strong."], ["Psalm 56:3", "What time I am afraid, I will trust in thee."], ["Proverbs 15:22", "Without counsel purposes are disappointed: but in the multitude of counsellors they are established."], ["Numbers 6:24-26", "The LORD bless thee, and keep thee: The LORD make his face shine upon thee, and be gracious unto thee: The LORD lift up his countenance upon thee, and give thee peace."], ["Psalm 90:17", "And let the beauty of the LORD our God be upon us: and establish thou the work of our hands upon us; yea, the work of our hands establish thou it."], ["Micah 6:8", "He hath shewed thee, O man, what is good; and what doth the LORD require of thee, but to do justly, and to love mercy, and to walk humbly with thy God?"], ["Proverbs 11:3", "The integrity of the upright shall guide them: but the perverseness of transgressors shall destroy them."], ["Romans 15:13", "Now the God of hope fill you with all joy and peace in believing, that ye may abound in hope, through the power of the Holy Ghost."], ["Psalm 32:8", "I will instruct thee and teach thee in the way which thou shalt go: I will guide thee with mine eye."], ["Deuteronomy 31:6", "Be strong and of a good courage, fear not, nor be afraid of them: for the LORD thy God, he it is that doth go with thee; he will not fail thee, nor forsake thee."], ["Proverbs 14:23", "In all labour there is profit: but the talk of the lips tendeth only to penury."], ["Psalm 62:8", "Trust in him at all times; ye people, pour out your heart before him: God is a refuge for us."], ["1 Peter 5:7", "Casting all your care upon him; for he careth for you."], ["Philippians 4:6", "Be careful for nothing; but in every thing by prayer and supplication with thanksgiving let your requests be made known unto God."], ["Proverbs 13:11", "Wealth gotten by vanity shall be diminished: but he that gathereth by labour shall increase."], ["Isaiah 26:3", "Thou wilt keep him in perfect peace, whose mind is stayed on thee: because he trusteth in thee."], ["Psalm 16:8", "I have set the LORD always before me: because he is at my right hand, I shall not be moved."]];
  function get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function bg() { var v = get('t58-bg'); return BGS.indexOf(v) >= 0 ? v : 'mountains'; }
  function verseOn() { return get('t58-verse') === '1'; }
  function applyBg() { document.documentElement.setAttribute('data-bg', bg()); }
  // Ease the new scene in (theme.css: html.t58-bg-swap) when the backdrop or the theme changes -- never on a plain page load.
  var swapTimer = 0;
  function swap() {
    var el = document.documentElement;
    el.classList.remove('t58-bg-swap');
    void el.offsetWidth; // restart the animation if it is already running
    el.classList.add('t58-bg-swap');
    clearTimeout(swapTimer);
    swapTimer = setTimeout(function () { el.classList.remove('t58-bg-swap'); }, 1000);
  }
  function slotVerse() { return VERSES[Math.floor(Date.now() / SLOT_MS) % VERSES.length]; }

  function renderVerse() {
    var existing = document.getElementById('t58-verse');
    if (!verseOn()) { if (existing) existing.remove(); return; }
    var host = document.querySelector('.t58-main');
    if (!host) return;
    var v = slotVerse();
    if (!existing) {
      existing = document.createElement('div');
      existing.id = 't58-verse'; existing.className = 't58-verse'; existing.setAttribute('role', 'note');
      host.insertBefore(existing, host.firstChild);
    }
    if (existing.dataset.ref === v[0]) return;
    existing.dataset.ref = v[0];
    existing.textContent = '';
    var q = document.createElement('span'); q.className = 't58-verse-text'; q.textContent = '\u201C' + v[1] + '\u201D';
    var r = document.createElement('span'); r.className = 't58-verse-ref'; r.textContent = '\u2014 ' + v[0] + ' (KJV)';
    existing.appendChild(q); existing.appendChild(r);
  }

  window.T58Personalize = {
    backgrounds: BGS, bg: bg, verse: verseOn,
    setBg: function (name) { if (BGS.indexOf(name) < 0) name = 'mountains'; set('t58-bg', name); applyBg(); swap(); },
    setVerse: function (on) { set('t58-verse', on ? '1' : '0'); renderVerse(); },
    verses: VERSES.length
  };

  applyBg();
  // The theme toggle (t58-chrome.js) flips data-theme, which swaps night <-> day scenes: fade that too.
  if (window.MutationObserver) {
    new MutationObserver(swap).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
  // Another tab changed the backdrop: follow it.
  window.addEventListener('storage', function (e) { if (e.key === 't58-bg') { applyBg(); swap(); } });
  function boot() { renderVerse(); setInterval(renderVerse, 60 * 1000); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
