# dataquest_arcade.py
# DataQuest: Infinite Structures - Arcade Edition (Pygame)
# Single-file, self-contained. Requires pygame: pip install pygame

import pygame, random, math, sys, time
from collections import deque

pygame.init()
WIDTH, HEIGHT = 1000, 700
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("consolas", 18)
BIGFONT = pygame.font.SysFont("consolas", 36)
TITLE_FONT = pygame.font.SysFont("consolas", 48, bold=True)

# Colors
WHITE = (245,245,245)
BLACK = (20,20,20)
GRAY = (180,180,180)
LIGHT = (220,240,255)
ACCENT = (80,160,220)
GOOD = (80,200,120)
BAD = (220,90,90)
GOLD = (235,190,80)
DARK = (25,40,60)

# Utility UI helpers
def draw_text(surface, txt, pos, font=FONT, color=BLACK):
    surface.blit(font.render(txt, True, color), pos)

def center_text(surface, txt, y, font=BIGFONT, color=BLACK):
    r = font.render(txt, True, color)
    surface.blit(r, (surface.get_width()//2 - r.get_width()//2, y))

def button(surface, rect, label, mouse_pos, clicked, bg=ACCENT):
    x,y,w,h = rect
    mx,my = mouse_pos
    hover = x<mx<x+w and y<my<y+h
    col = tuple(min(255, c+20) for c in bg) if hover else bg
    pygame.draw.rect(surface, col, rect, border_radius=10)
    pygame.draw.rect(surface, DARK, rect, 3, border_radius=10)
    draw_text(surface, label, (x+12, y+10), FONT, BLACK)
    if hover and clicked:
        return True
    return False

# Simple animated sparkle
def sparkle(surface, pos, t):
    x,y = pos
    r = 3 + int(2*math.sin(t*10))
    pygame.draw.circle(surface, GOLD, (int(x), int(y)), r)

# Game control and state
class GameState:
    def __init__(self):
        self.score = 0
        self.unlocked = set()
        self.current = None
        self.running = True
        self.mode = "hub"  # hub, play
        self.hint_tokens = 3
        self.last_msg = ("", 0)  # (text, expiry_time)

    def show_msg(self, text, ttl=2.5):
        self.last_msg = (text, time.time()+ttl)

game = GameState()

# Mini-game base class
class MiniGame:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        self.finished = False
        self.reward = 10

    def start(self):
        pass
    def handle_event(self, e):
        pass
    def update(self, dt):
        pass
    def draw(self, surface):
        pass
    def hint(self):
        return "No hint available."
    def explanation(self):
        return "No explanation specified."

# ---------- MiniGames Implementations ----------
# 1) Array Swap Puzzle (drag to swap)
class ArraySwap(MiniGame):
    def __init__(self):
        super().__init__("Array Shuffle", "Sort the array by swapping elements.")
    def start(self):
        n = random.randint(6,9)
        self.arr = list(range(1, n+1))
        random.shuffle(self.arr)
        self.target = sorted(self.arr)
        self.tiles = []
        w = 70; h = 70; gap = 12
        total_w = n * w + (n-1)*gap
        sx = WIDTH//2 - total_w//2
        y = HEIGHT//2 - 80
        for i,v in enumerate(self.arr):
            rect = pygame.Rect(sx + i*(w+gap), y, w, h)
            self.tiles.append({"v":v, "rect":rect, "offset":(0,0)})
        self.dragging = None
        self.anim = []
        self.moves = 0
        self.start_time = time.time()
        self.reward = 20
    def handle_event(self, e):
        if e.type==pygame.MOUSEBUTTONDOWN:
            for i,t in enumerate(self.tiles):
                if t["rect"].collidepoint(e.pos):
                    self.dragging = (i, (e.pos[0]-t["rect"].x, e.pos[1]-t["rect"].y))
                    break
        if e.type==pygame.MOUSEBUTTONUP and self.dragging is not None:
            i,off = self.dragging
            self.dragging=None
            # check if overlapping another tile -> swap
            for j,t in enumerate(self.tiles):
                if j!=i and t["rect"].collidepoint(pygame.mouse.get_pos()):
                    self.tiles[i]["rect"], self.tiles[j]["rect"] = self.tiles[j]["rect"], self.tiles[i]["rect"]
                    self.tiles[i], self.tiles[j] = self.tiles[j], self.tiles[i]  # swap objects
                    self.moves += 1
                    game.show_msg("Swap!")
                    break
    def update(self, dt):
        if self.dragging:
            i, (ox,oy) = self.dragging
            mx,my = pygame.mouse.get_pos()
            self.tiles[i]["rect"].x = mx-ox
            self.tiles[i]["rect"].y = my-oy
        # check win by order of values left->right
        order = [t["v"] for t in sorted(self.tiles, key=lambda k:k["rect"].x)]
        if order==self.target and not self.finished:
            self.finished = True
            elapsed = time.time()-self.start_time
            pts = max(10, int(self.reward - elapsed - self.moves*1.5))
            game.score += pts
            game.unlocked.add("array")
            game.show_msg(f"Sorted! +{pts} points")
    def draw(self, s):
        center_text(s, "Array Shuffle", 30, TITLE_FONT)
        draw_text(s, self.desc, (40,80), FONT, DARK)
        # draw target hint
        draw_text(s, "Target: " + " ".join(map(str,self.target)), (40,110), FONT, GRAY)
        for t in self.tiles:
            r = t["rect"]
            pygame.draw.rect(s, LIGHT, r, border_radius=8)
            pygame.draw.rect(s, DARK, r, 3, border_radius=8)
            draw_text(s, str(t["v"]), (r.x+24, r.y+20), BIGFONT, DARK)
        draw_text(s, f"Moves: {self.moves}  Score: {game.score}", (40,HEIGHT-40), FONT, BLACK)
    def hint(self):
        # simple hint: find first mismatch
        order = [t["v"] for t in sorted(self.tiles, key=lambda k:k["rect"].x)]
        for i,(a,b) in enumerate(zip(order,self.target)):
            if a!=b:
                j = order.index(b)
                return f"Swap position {i+1} with {j+1} (1-based)."
        return "Try looking for smallest element in front."
    def explanation(self):
        return ("Arrays are contiguous sequences. Access by index is O(1). "
                "Sorting reorders elements; swapping changes positions; insert/delete in middle is O(n).")

# 2) Linked List Repair (connect nodes)
class LinkedRepair(MiniGame):
    def __init__(self):
        super().__init__("Linked Repair", "Reconnect the broken pointer to restore order.")
    def start(self):
        size = random.randint(5,7)
        nodes = list(range(1, size+1))
        random.shuffle(nodes)
        # create correct order randomly
        order = nodes[:]
        random.shuffle(order)
        # head is order[0]
        self.head = order[0]
        self.next = {}
        for i in range(len(order)-1):
            self.next[order[i]] = order[i+1]
        self.next[order[-1]] = None
        # break one
        self.broken = random.choice(order[:-1])
        self.saved = self.next[self.broken]
        self.next[self.broken] = random.choice(order + [None])
        # position nodes in circle
        cx,cy = WIDTH//2, HEIGHT//2 + 10
        R = 180
        self.node_pos = {}
        for i,v in enumerate(order):
            ang = i*2*math.pi/len(order) - math.pi/2
            x = cx + int(R*math.cos(ang))
            y = cy + int(R*math.sin(ang))
            self.node_pos[v] = pygame.Rect(x-28,y-28,56,56)
        self.selected = None
        self.attempts = 0
        self.reward = 18
    def handle_event(self, e):
        if e.type==pygame.MOUSEBUTTONDOWN:
            for v,rect in self.node_pos.items():
                if rect.collidepoint(e.pos):
                    if self.selected is None:
                        self.selected = v
                        game.show_msg(f"Selected node {v}")
                    else:
                        # connect selected -> v (or None if clicked same)
                        target = v
                        if target==self.selected:
                            # treat as setting to None
                            self.next[self.selected] = None
                        else:
                            self.next[self.selected] = target
                        self.attempts += 1
                        self.selected = None
    def update(self, dt):
        # check traversal
        seen = set()
        cur = self.head
        while cur is not None:
            if cur in seen: break
            seen.add(cur)
            cur = self.next.get(cur, None)
        if len(seen) == len(self.node_pos) and not self.finished:
            self.finished = True
            pts = max(8, self.reward - self.attempts*2)
            game.score += pts
            game.unlocked.add("linked")
            game.show_msg(f"List repaired! +{pts} pts")
    def draw(self, s):
        center_text(s, "Linked List Repair", 30, TITLE_FONT)
        draw_text(s, self.desc, (40,80), FONT, DARK)
        # draw nodes and arrows
        for v,rect in self.node_pos.items():
            color = ACCENT if v!=self.broken else BAD
            pygame.draw.ellipse(s, LIGHT, rect)
            pygame.draw.ellipse(s, DARK, rect, 3)
            draw_text(s, str(v), (rect.x+18, rect.y+18), BIGFONT, DARK)
        # draw arrows showing current next
        for v,rect in self.node_pos.items():
            tgt = self.next.get(v, None)
            if tgt is None: continue
            r2 = self.node_pos[tgt]
            sx = rect.centerx; sy = rect.centery
            ex = r2.centerx; ey = r2.centery
            # draw curved arrow
            draw_arrow(s, (sx,sy), (ex,ey))
        # highlight selected
        if self.selected:
            r = self.node_pos[self.selected]
            pygame.draw.ellipse(s, (255,255,200), r, 4)
        draw_text(s, f"Attempts: {self.attempts}  Score: {game.score}", (40,HEIGHT-40), FONT, BLACK)

    def hint(self):
        # hint: check node not in traversal
        seen = set()
        cur = self.head
        while cur is not None:
            if cur in seen: break
            seen.add(cur)
            cur = self.next.get(cur, None)
        missing = [v for v in self.node_pos if v not in seen]
        if missing:
            return f"Nodes not reached in traversal: {missing}. Connect the tail to missing nodes."
        return "Try checking for a cycle or None pointer."

    def explanation(self):
        return ("A linked list uses nodes with pointers. Insertion/deletion at a known node is O(1); "
                "search is O(n). Pointers are the building blocks of dynamic structures.")

# draw arrow helper
def draw_arrow(surface, a, b, color=DARK):
    ax,ay = a; bx,by = b
    dx = bx-ax; dy = by-ay
    dist = math.hypot(dx,dy)
    if dist<1: return
    ux,uy = dx/dist, dy/dist
    # start slightly out of source radius, end slightly before target
    start = (ax + ux*38, ay + uy*38)
    end = (bx - ux*38, by - uy*38)
    pygame.draw.line(surface, color, start, end, 4)
    # arrowhead
    ang = math.atan2(uy, ux)
    left = (end[0] - 12*math.cos(ang-0.5), end[1] - 12*math.sin(ang-0.5))
    right= (end[0] - 12*math.cos(ang+0.5), end[1] - 12*math.sin(ang+0.5))
    pygame.draw.polygon(surface, color, [end, left, right])

# 3) Stack Sequence (buttons to push/pop to produce target)
class StackPuzzle(MiniGame):
    def __init__(self):
        super().__init__("Stack Master", "Use push/pop to transform input into the target (LIFO).")
    def start(self):
        n = random.randint(4,6)
        self.sequence = list(range(1,n+1))
        random.shuffle(self.sequence)
        # target choose reverse or shuffled
        self.target = self.sequence[::-1]
        self.stack = []
        self.out = []
        self.idx = 0
        self.reward = 14
        self.start_time = time.time()
    def handle_event(self, e):
        if e.type==pygame.MOUSEBUTTONDOWN:
            mx,my = e.pos
            if 820<mx<960 and 220<my<270:
                # push
                if self.idx < len(self.sequence):
                    self.stack.append(self.sequence[self.idx]); self.idx += 1
            if 820<mx<960 and 300<my<350:
                # pop
                if self.stack:
                    self.out.append(self.stack.pop())
    def update(self, dt):
        if self.out==self.target and not self.finished:
            self.finished = True
            elapsed = time.time()-self.start_time
            pts = max(10, int(self.reward - elapsed/2 - len(self.out)))
            game.score += pts
            game.unlocked.add("stack")
            game.show_msg(f"Stack goal reached! +{pts} pts")
    def draw(self, s):
        center_text(s, "Stack Master", 30, TITLE_FONT)
        draw_text(s, self.desc, (40,80), FONT, DARK)
        # input sequence
        draw_text(s, "Input sequence (next to push):", (40,120), FONT, GRAY)
        for i,v in enumerate(self.sequence):
            color = LIGHT if i>=self.idx else (200,220,240)
            pygame.draw.rect(s, color, (40 + i*60, 150, 48,48), border_radius=8)
            draw_text(s, str(v), (52 + i*60, 162), FONT, DARK)
        # stack display
        pygame.draw.rect(s, (245,245,245), (380,140,300,340), border_radius=10)
        pygame.draw.rect(s, DARK, (380,140,300,340), 3, border_radius=10)
        draw_text(s, "Stack (top at top)", (400,150), FONT, DARK)
        for i,v in enumerate(reversed(self.stack)):
            pygame.draw.rect(s, (210,230,245), (420,190 + i*48, 200, 42), border_radius=6)
            draw_text(s, str(v), (530-16, 198 + i*48), FONT, DARK)
        # output area
        draw_text(s, "Output:", (40, 220), FONT, DARK)
        draw_text(s, " ".join(map(str,self.out)), (40,250), BIGFONT, DARK)
        # buttons
        pygame.draw.rect(s, ACCENT, (820,220,140,50), border_radius=12)
        draw_text(s, "PUSH", (850,235), BIGFONT, BLACK)
        pygame.draw.rect(s, ACCENT, (820,300,140,50), border_radius=12)
        draw_text(s, "POP", (860,315), BIGFONT, BLACK)
        draw_text(s, f"Score: {game.score}", (40, HEIGHT-40), FONT, BLACK)
    def hint(self):
        return "Stacks are LIFO: to get reversed sequence, push all then pop all."

    def explanation(self):
        return ("Stack: Last-In-First-Out. push() adds to top; pop() removes top. Used in recursion, backtracking, expression eval.")

# 4) Binary Tree Traversal (click nodes in requested order)
class TreeTraversal(MiniGame):
    def __init__(self):
        super().__init__("Tree Traversal", "Click nodes in the requested traversal order.")
    def start(self):
        # create balanced BST with 7 nodes
        vals = random.sample(range(1,99), 7)
        vals.sort()
        self.nodes = []
        def build(arr, depth, x, y, span):
            if not arr: return None
            m = len(arr)//2
            v = arr[m]
            idx = len(self.nodes)
            rect = pygame.Rect(x-28, y-18, 56,36)
            self.nodes.append({"v":v,"rect":rect})
            left = build(arr[:m], depth+1, x-span, y+90, span//2)
            right= build(arr[m+1:], depth+1, x+span, y+90, span//2)
            return idx
        build(vals, 0, WIDTH//2, 140, 260)
        self.order = random.choice(["in","pre","post"])
        # compute traversal
        def trav(arr, typ):
            if not arr: return []
            m = len(arr)//2
            left = trav(arr[:m], typ)
            right= trav(arr[m+1:], typ)
            v = [arr[m]]
            if typ=="in": return left + v + right
            if typ=="pre": return v + left + right
            return left + right + v
        self.target = trav(vals, self.order)
        self.clicked = []
        self.reward = 20
        self.start_time = time.time()
    def handle_event(self, e):
        if e.type==pygame.MOUSEBUTTONDOWN:
            for nd in self.nodes:
                if nd["rect"].collidepoint(e.pos):
                    if nd["v"] not in self.clicked:
                        self.clicked.append(nd["v"])
    def update(self, dt):
        if self.clicked==self.target and not self.finished:
            self.finished = True
            pts = max(12, self.reward - int((time.time()-self.start_time)))
            game.score += pts
            game.unlocked.add("tree")
            game.show_msg(f"Traversal correct! +{pts} pts")
    def draw(self, s):
        center_text(s, "Tree Traversal", 30, TITLE_FONT)
        draw_text(s, f"Click nodes in {self.order}-order.", (40,80), FONT, DARK)
        # compute positions from rect centers in nodes
        for nd in self.nodes:
            c = nd["rect"].center
            # draw connections approximate by vertical lines (for clarity)
        # draw nodes and simple vertical layout
        # arrange using y positions saved in rect
        for nd in self.nodes:
            r = nd["rect"]
            pygame.draw.rect(s, LIGHT, r, border_radius=6)
            pygame.draw.rect(s, DARK, r, 2, border_radius=6)
            draw_text(s, str(nd["v"]), (r.x+12, r.y+6), FONT, DARK)
            if nd["v"] in self.clicked:
                pygame.draw.circle(s, GOOD, (r.right-8, r.y+8), 6)
        draw_text(s, "Clicked: " + " ".join(map(str,self.clicked)), (40, HEIGHT-100), FONT, BLACK)
        draw_text(s, f"Target length: {len(self.target)}  Score: {game.score}", (40, HEIGHT-40), FONT, BLACK)

    def hint(self):
        return "For inorder: left-root-right. Preorder: root-left-right. Postorder: left-right-root."

    def explanation(self):
        return ("Tree traversals visit nodes in different orders. Inorder yields sorted keys for BST. Traversals are fundamental to many tree algorithms.")

# 5) Graph BFS Shortest Path (click nodes to form path)
class GraphBFS(MiniGame):
    def __init__(self):
        super().__init__("Graph Pathfinder", "Find the shortest path between two nodes (BFS).")
    def start(self):
        n = random.randint(6,8)
        self.nodes = list(range(1, n+1))
        # place nodes randomly but not overlapping
        self.node_rects = {}
        for v in self.nodes:
            while True:
                x = random.randint(120, WIDTH-200)
                y = random.randint(140, HEIGHT-160)
                r = pygame.Rect(x-24, y-24, 48, 48)
                if all(not r.colliderect(o) for o in self.node_rects.values()):
                    self.node_rects[v]=r
                    break
        # create connected graph edges
        edges = []
        for i in range(2, n+1):
            j = random.randint(1, i-1)
            edges.append((i,j))
        for _ in range(n//2):
            a,b = random.sample(self.nodes,2)
            if (a,b) not in edges and (b,a) not in edges:
                edges.append((a,b))
        self.adj = {v:set() for v in self.nodes}
        for a,b in edges:
            self.adj[a].add(b); self.adj[b].add(a)
        self.s, self.t = random.sample(self.nodes,2)
        self.selected = []
        self.reward = 25
        # compute BFS shortest length
        q = deque([self.s]); parent={self.s:None}
        while q:
            u = q.popleft()
            if u==self.t: break
            for v in self.adj[u]:
                if v not in parent:
                    parent[v]=u; q.append(v)
        # reconstruct
        cur = self.t; path=[]
        if cur in parent:
            while cur is not None:
                path.append(cur); cur=parent[cur]
            path.reverse()
        self.shortest = path
    def handle_event(self, e):
        if e.type==pygame.MOUSEBUTTONDOWN:
            for v,r in self.node_rects.items():
                if r.collidepoint(e.pos):
                    if not self.selected:
                        if v==self.s:
                            self.selected.append(v)
                    else:
                        # allow append only if neighbor of last
                        if v in self.adj[self.selected[-1]]:
                            self.selected.append(v)
    def update(self, dt):
        if self.selected and self.selected[0]==self.s and self.selected[-1]==self.t:
            # check if path valid
            if len(self.selected)-1 == len(self.shortest)-1:
                if self.selected==self.shortest and not self.finished:
                    self.finished=True
                    game.score += self.reward
                    game.unlocked.add("graph")
                    game.show_msg(f"Shortest path found! +{self.reward} pts")
            else:
                # valid but not shortest
                game.show_msg("Valid path but not shortest.", ttl=1.2)
    def draw(self, s):
        center_text(s, "Graph Pathfinder", 30, TITLE_FONT)
        draw_text(s, f"Connect nodes from {self.s} to {self.t} by clicking neighbors.", (40,80), FONT, DARK)
        # draw edges
        for a in self.nodes:
            for b in self.adj[a]:
                if a < b:
                    ax,ay = self.node_rects[a].center; bx,by = self.node_rects[b].center
                    pygame.draw.line(s, GRAY, (ax,ay),(bx,by), 3)
        # highlight shortest path (ghost)
        for i in range(len(self.shortest)-1):
            a=self.shortest[i]; b=self.shortest[i+1]
            ax,ay=self.node_rects[a].center; bx,by=self.node_rects[b].center
            pygame.draw.line(s, (200,230,200), (ax,ay),(bx,by), 6)
        # draw nodes
        for v,r in self.node_rects.items():
            col = LIGHT
            if v==self.s: col=(200,240,255)
            if v==self.t: col=(255,230,230)
            pygame.draw.ellipse(s, col, r)
            pygame.draw.ellipse(s, DARK, r,2)
            draw_text(s, str(v), (r.x+18, r.y+10), FONT, DARK)
        # draw selected path
        for i in range(len(self.selected)-1):
            a=self.selected[i]; b=self.selected[i+1]
            ax,ay=self.node_rects[a].center; bx,by=self.node_rects[b].center
            pygame.draw.line(s, ACCENT, (ax,ay),(bx,by), 5)
        draw_text(s, f"Selected: {'->'.join(map(str,self.selected))}", (40, HEIGHT-80), FONT, BLACK)
        draw_text(s, f"Score: {game.score}", (40, HEIGHT-40), FONT, BLACK)
    def hint(self):
        if not self.shortest:
            return "No path exists (unlikely)."
        return f"Shortest path length = {len(self.shortest)-1}. Try BFS from {self.s}."

    def explanation(self):
        return ("BFS finds shortest path in unweighted graphs by exploring in layers. "
                "BFS uses a queue and marks visited nodes; complexity O(V+E).")

# Mini-game registry & helper to pick random
MINI_LIST = [ArraySwap, LinkedRepair, StackPuzzle, TreeTraversal, GraphBFS]

def pick_random_game():
    cls = random.choice(MINI_LIST)
    g = cls()
    g.start()
    return g

# ---------- Hub / UI screens ----------
def hub_screen(surface, mouse_pos, clicked):
    surface.fill((245,250,255))
    center_text(surface, "DataQuest: Infinite Structures", 36, TITLE_FONT, DARK)
    center_text(surface, "Arcade Edition", 96, BIGFONT, ACCENT)
    draw_text(surface, "Score: " + str(game.score), (40,40), FONT, DARK)
    # buttons
    if button(surface, (80,160,260,70), "Random Challenge", mouse_pos, clicked):
        game.current = pick_random_game()
        game.mode = "play"
    if button(surface, (380,160,260,70), "Pick Challenge", mouse_pos, clicked):
        game.mode = "pick"
    if button(surface, (680,160,260,70), "Practice Mode", mouse_pos, clicked):
        game.current = pick_random_game()
        game.mode = "play"
    # show unlocked badges
    draw_text(surface, "Unlocked: " + ", ".join(sorted(game.unlocked)) if game.unlocked else "Unlocked: (none)", (40,260), FONT, DARK)
    # quick tutorial
    draw_text(surface, "How to play: Select a challenge, follow on-screen controls. Use hints (H) if stuck.", (40,320), FONT, DARK)
    draw_text(surface, "Get points to unlock concept cards with code snippets.", (40,350), FONT, DARK)
    # footer
    draw_text(surface, "Tip: The game is endless. Have fun learning!", (40, HEIGHT-40), FONT, DARK)

def pick_screen(surface, mouse_pos, clicked):
    surface.fill((255,255,250))
    center_text(surface, "Choose a Challenge", 20, TITLE_FONT, DARK)
    for i,cls in enumerate(MINI_LIST):
        y = 130 + i*90
        name = cls().__class__.__name__ if False else cls().name
        desc = cls().desc if hasattr(cls(), "desc") else ""
        rect = (80, y, 840, 70)
        pygame.draw.rect(surface, (250,250,255), rect, border_radius=10)
        pygame.draw.rect(surface, DARK, rect, 2, border_radius=10)
        draw_text(surface, name, (100, y+8), BIGFONT, DARK)
        draw_text(surface, desc, (100, y+44), FONT, GRAY)
        if button(surface, (820, y+12, 80, 46), "Play", mouse_pos, clicked):
            game.current = cls(); game.current.start(); game.mode = "play"

# In-game HUD controls
def in_game_hud(surface, current_game):
    # top right help/hint buttons
    mx, my = pygame.mouse.get_pos()
    # Hint button (press H too)
    pygame.draw.rect(surface, (238,238,245), (780,20,200,44), border_radius=10)
    pygame.draw.rect(surface, DARK, (780,20,200,44), 2, border_radius=10)
    draw_text(surface, f"Hint [{game.hint_tokens}]: Click or press H", (800,30), FONT, DARK)
    # Explanation / cheat card
    pygame.draw.rect(surface, (238,238,245), (780,72,200,44), border_radius=10)
    pygame.draw.rect(surface, DARK, (780,72,200,44), 2, border_radius=10)
    draw_text(surface, "Show Concept (C)", (800,80), FONT, DARK)
    # back button
    if button(surface, (40,20,120,44), "Back", pygame.mouse.get_pos(), False, bg=(200,200,200)):
        pass

# Main loop
def main_loop():
    last_click = False
    while game.running:
        dt = CLOCK.tick(60)/1000.0
        clicked = False
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                game.running=False
            if e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE:
                    if game.mode=="play":
                        game.mode="hub"; game.current=None
                    else:
                        game.running=False
                if e.key==pygame.K_h:
                    # hint
                    if game.hint_tokens>0 and game.current:
                        game.hint_tokens-=1
                        game.show_msg("HINT: " + game.current.hint(), ttl=3.0)
                if e.key==pygame.K_c and game.current:
                    game.show_msg("CONCEPT: " + game.current.explanation(), ttl=4.0)
            if e.type==pygame.MOUSEBUTTONDOWN:
                last_click = True
                clicked = True
                if game.mode=="hub":
                    # hub buttons handle via hub_screen's button detection: we pass clicked boolean next frame
                    pass
                elif game.mode=="pick":
                    pass
                elif game.mode=="play" and game.current:
                    # forward event to game
                    game.current.handle_event(e)
            if e.type==pygame.MOUSEBUTTONUP:
                last_click=False

        # Screen updates
        if game.mode=="hub":
            SCREEN.fill(WHITE)
            # pass clicked detection to button function by drawing UI with last click flag
            hub_screen(SCREEN, pygame.mouse.get_pos(), clicked)
        elif game.mode=="pick":
            pick_screen(SCREEN, pygame.mouse.get_pos(), clicked)
        elif game.mode=="play":
            SCREEN.fill(WHITE)
            if game.current:
                game.current.update(dt)
                game.current.draw(SCREEN)
                # HUD hints
                in_game_hud(SCREEN, game.current)
                # persistent message
        # draw floating message
        if game.last_msg[1] > time.time():
            txt = game.last_msg[0]
            r = FONT.render(txt, True, BLACK)
            SCREEN.blit(r, (WIDTH//2 - r.get_width()//2, HEIGHT-80))
            sparkle(SCREEN, (WIDTH//2 + r.get_width()//2 + 20, HEIGHT-76), time.time())
        # click handling for hub buttons (two-stage because button() expects clicked flag)
        if game.mode=="hub" and clicked:
            # recreate hub to detect button click; this time treat clicked True
            hub_screen(SCREEN, pygame.mouse.get_pos(), True)
        if game.mode=="pick" and clicked:
            pick_screen(SCREEN, pygame.mouse.get_pos(), True)

        pygame.display.flip()

# Start screen intro animation
def intro():
    t0 = time.time()
    while True:
        dt = CLOCK.tick(60)/1000.0
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type==pygame.KEYDOWN or (e.type==pygame.MOUSEBUTTONDOWN):
                return
        SCREEN.fill((18,28,40))
        center_text(SCREEN, "DataQuest", 120, TITLE_FONT, (200,230,255))
        center_text(SCREEN, "Infinite Structures — Arcade Edition", 190, BIGFONT, (160,200,230))
        draw_text(SCREEN, "Click or press any key to begin...", (WIDTH//2-160, HEIGHT-120), FONT, GRAY)
        # animated orbs
        for i in range(12):
            x = WIDTH//2 + math.cos(time.time()*0.8 + i)*240*math.sin(i)
            y = HEIGHT//2 + math.sin(time.time()*0.6 + i)*40
            pygame.draw.circle(SCREEN, (40+i*10, 80+i*8, 160), (int(WIDTH//2 + math.cos(time.time()*0.5+i)*200), int(HEIGHT//2 + math.sin(time.time()*0.4+i)*60)), 8)
        pygame.display.flip()

# Run
if __name__=="__main__":
    try:
        intro()
        main_loop()
    except Exception as ex:
        # show error on screen
        SCREEN.fill((220,20,20))
        center_text(SCREEN, "An unexpected error occurred.", 200, BIGFONT, WHITE)
        draw_text(SCREEN, str(ex), (40,260), FONT, WHITE)
        pygame.display.flip()
        pygame.time.wait(4000)
    finally:
        pygame.quit()