document.addEventListener("DOMContentLoaded", () => {
    const toc = document.querySelector(".bd-toc-nav");
    if (!toc) return;

    // Get all TOC links
    const links = toc.querySelectorAll("a");

    const seen = new Set();
    links.forEach(link => {
      let text = link.textContent.trim();
      let norm = text.replace(/\(\)$/, ""); // strip trailing ()

      if (seen.has(norm)) {
        // hide duplicate
        link.parentElement.style.display = "none";
      } else {
        seen.add(norm);
      }
    });
  });

document.addEventListener("DOMContentLoaded", () => {
  const toc = document.querySelector(".toctree-wrapper");
  if (!toc) return;

  const leaf_traversal_fn = (leaf) => {
    try {
      const links = leaf.querySelectorAll("a");
      if (!links) return;
      const seen = new Set();
      links.forEach(link => {
        let text = link.textContent.trim();
        let norm = text.replace(/\(\)$/, ""); // strip trailing ()

        if (seen.has(norm)) {
          // hide duplicate
          link.parentElement.style.display = "none";
        } else {
          seen.add(norm);
        }
      });
    } catch (error) {
      console.error(error);
    }
  };
  queue = [toc.querySelector("li")];
  while (queue.length > 0) {
    const tree = queue.shift();
    try {
      if (tree.childElementCount > 1 &&
          tree.firstChild.hasAttribute("href") &&
          tree.firstChild.attributes["href"].value.includes("#")) {
        leaf_traversal_fn(tree.childNodes[1]);
      } else {
        const child = tree.querySelector("li");
        if (child) {
          queue.push(child);
        }
      }
    } catch (error) {
      console.error(error);
    }
    if (tree.nextElementSibling) {
      queue.push(tree.nextElementSibling);
    }
  }
  });
