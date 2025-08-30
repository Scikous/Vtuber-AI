import logging
import time
from typing import Dict, Any

try:
    from evdev import UInput, ecodes as e
except ImportError:
    print("The 'evdev' library is not installed. Please install it with 'pip install evdev'")
    print("This module is intended for Linux systems only.")
    exit(1)

try:
    from screeninfo import get_monitors
except ImportError:
    print("The 'screeninfo' library is not installed. Please install it with 'pip install screeninfo'")
    exit(1)

class InputController:
    """
    Emulates a standard mouse and keyboard using a purely relative evdev device.
    This is the most compatible method for Wayland compositors.
    It maintains an internal state to translate absolute coordinates into
    the necessary relative movements for the mouse.
    """

    def __init__(self):
        """Initializes the virtual input device."""
        self.screen_width, self.screen_height = self._get_screen_resolution()
        
        # Internal state for tracking the virtual cursor's position.
        # We initialize it to the center of the screen.
        self.current_x = self.screen_width // 2
        self.current_y = self.screen_height // 2
        
        self.ui = None
        try:
            # Emulate a standard mouse (EV_REL) and a full keyboard (EV_KEY).
            # This combination is highly compatible with modern desktop environments.
            capabilities = {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE] + list(range(e.KEY_RESERVED, e.KEY_MAX + 1)),
                e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL],
            }

            self.ui = UInput(capabilities, name='AI-Agent-Controller', version=0x1)
            logging.info(f"InputController initialized. Internal cursor at: ({self.current_x}, {self.current_y})")

        except Exception as err:
            logging.error(
                f"Failed to initialize UInput device: {err}\n"
                "Please ensure you have the correct permissions for /dev/uinput.\n"
                "Did you run the one-time setup and reboot?"
            )
            raise

    def _get_screen_resolution(self) -> tuple[int, int]:
        """Detects the primary monitor's resolution."""
        try:
            monitors = get_monitors()
            primary_monitor = next((m for m in monitors if m.is_primary), monitors[0])
            return primary_monitor.width, primary_monitor.height
        except Exception:
            logging.warning("Could not auto-detect screen resolution. Falling back to 1920x1080.")
            return 1920, 1080

    def close(self):
        """Closes and destroys the virtual input device."""
        if self.ui:
            self.ui.close()
            logging.info("UInput device closed.")

    def execute_action(self, action: Dict[str, Any]):
        """Public method to dispatch an action to the appropriate handler."""
        if not self.ui:
            logging.error("Cannot execute action, UInput device is not initialized.")
            return

        action_type = action.get("type")
        details = action.get("details", {})

        handler_map = {
            "mouse_move": self._handle_mouse_move,
            "mouse_click": self._handle_mouse_click,
            "click": self._handle_mouse_click, # Alias
            "key_press": self._handle_key_press,
            "key_down": self._handle_key_down,
            "key_up": self._handle_key_up,
        }
        
        handler_method = handler_map.get(action_type)

        if handler_method:
            try:
                handler_method(details)
            except Exception as err:
                logging.error(f"Error executing action '{action_type}': {err}", exc_info=True)
        else:
            logging.warning(f"No handler found for action type: '{action_type}'")

    def _handle_mouse_move(self, details: Dict[str, Any]):
        """Translates absolute coordinates into relative movement events."""
        target_x = details.get("target_x")
        target_y = details.get("target_y")
        if target_x is None or target_y is None: return

        target_x, target_y = int(target_x), int(target_y)

        delta_x = target_x - self.current_x
        delta_y = target_y - self.current_y

        if delta_x != 0:
            self.ui.write(e.EV_REL, e.REL_X, delta_x)
        if delta_y != 0:
            self.ui.write(e.EV_REL, e.REL_Y, delta_y)
        
        if delta_x != 0 or delta_y != 0:
            self.ui.syn()

        self.current_x = target_x
        self.current_y = target_y

    def _handle_mouse_click(self, details: Dict[str, Any]):
        """Performs a click, moving to a position first if specified."""
        if "target_x" in details and "target_y" in details:
            self._handle_mouse_move(details)
            time.sleep(0.03)

        button_str = details.get("button", "left").lower()
        button_map = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT, "middle": e.BTN_MIDDLE}
        button = button_map.get(button_str, e.BTN_LEFT)

        self.ui.write(e.EV_KEY, button, 1); self.ui.syn() # Press
        time.sleep(0.05)
        self.ui.write(e.EV_KEY, button, 0); self.ui.syn() # Release
        logging.info(f"Clicked {button_str} button at virtual coords ({self.current_x}, {self.current_y}).")

    def _get_keycode(self, key_str: str) -> int | None:
        """Maps a human-readable string to an evdev keycode."""
        if not key_str: return None
        
        key_str = key_str.upper()
        
        alias_map = {
            "CTRL": "LEFTCTRL", "CONTROL": "LEFTCTRL",
            "SHIFT": "LEFTSHIFT",
            "ALT": "LEFTALT",
            "WIN": "LEFTMETA", "SUPER": "LEFTMETA",
            "ENTER": "ENTER", "RETURN": "ENTER",
            "ESC": "ESC",
            "SPACE": "SPACE",
        }
        key_str = alias_map.get(key_str, key_str)
        
        keycode = getattr(e, f'KEY_{key_str}', None)
        if not keycode:
            logging.warning(f"Unknown key: '{key_str}'")
        return keycode

    def _handle_key_press(self, details: Dict[str, Any]):
        """Handles a single, quick key press (tap)."""
        key_code = self._get_keycode(details.get("key"))
        if not key_code: return
        
        self.ui.write(e.EV_KEY, key_code, 1); self.ui.syn() # Key down
        time.sleep(0.05)
        self.ui.write(e.EV_KEY, key_code, 0); self.ui.syn() # Key up

    def _handle_key_down(self, details: Dict[str, Any]):
        """Handles pressing and holding a key down."""
        key_code = self._get_keycode(details.get("key"))
        if not key_code: return
        
        self.ui.write(e.EV_KEY, key_code, 1); self.ui.syn()

    def _handle_key_up(self, details: Dict[str, Any]):
        """Handles releasing a key."""
        key_code = self._get_keycode(details.get("key"))
        if not key_code: return
            
        self.ui.write(e.EV_KEY, key_code, 0); self.ui.syn()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")
    
    controller = None
    try:
        controller = InputController()
        print("Controller initialized. The virtual cursor starts at screen center.")
        print("Testing in 3 seconds... Please open a text editor to see the keyboard test.")
        time.sleep(3)

        # --- Mouse Test ---
        print("--- Testing Mouse ---")
        print(f"Moving to (300, 400)...")
        controller.execute_action({"type": "mouse_move", "details": {"target_x": 300, "target_y": 400}})
        time.sleep(1)
        print(f"Clicking at current location...")
        controller.execute_action({"type": "mouse_click", "details": {}})
        time.sleep(1)

        # --- Keyboard Test ---
        print("\n--- Testing Keyboard ---")
        print("Typing: 'hello'")
        controller.execute_action({"type": "key_press", "details": {"key": "h"}})
        controller.execute_action({"type": "key_press", "details": {"key": "e"}})
        controller.execute_action({"type": "key_press", "details": {"key": "l"}})
        controller.execute_action({"type": "key_press", "details": {"key": "l"}})
        controller.execute_action({"type": "key_press", "details": {"key": "o"}})
        time.sleep(1)

        print("Typing: ' WORLD' (with shift modifier)")
        controller.execute_action({"type": "key_press", "details": {"key": "space"}})
        controller.execute_action({"type": "key_down", "details": {"key": "shift"}})
        controller.execute_action({"type": "key_press", "details": {"key": "w"}})
        controller.execute_action({"type": "key_press", "details": {"key": "o"}})
        controller.execute_action({"type": "key_press", "details": {"key": "r"}})
        controller.execute_action({"type": "key_press", "details": {"key": "l"}})
        controller.execute_action({"type": "key_press", "details": {"key": "d"}})
        controller.execute_action({"type": "key_up", "details": {"key": "shift"}})
        
        print("\nTests complete.")

    except Exception as main_err:
        logging.error(f"An error occurred during the test: {main_err}", exc_info=True)
    finally:
        if controller:
            controller.close()