import java.util.Scanner;

public class gcd_of_twonumbers {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter first number: ");
        int x = scanner.nextInt();

        System.out.print("Enter second number: ");

        int y = scanner.nextInt();
        int min = Math.min(x, y);

        int gcd = 1;

        for (int i = min; i >= 1; i--) {
            if (x % i == 0 && y % i == 0) {
                gcd = i;
                break;  // Stop at the first (largest) common divisor
            }
        }

        System.out.println("GCD is: " + gcd);
        scanner.close();
    }
}
